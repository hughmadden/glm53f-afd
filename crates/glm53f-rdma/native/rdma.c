// glm53f RDMA transport shim, from mimo26f-afd v1.2.0 (perf reset R2 part 2,
// mimo26f-afd docs/design/perf-reset-vs-ds41rt.md).
//
// The upstream DS41RT v15 design on libibverbs: one RC queue pair per peer, SEND
// into pre-posted RECV slots of registered buffers, completions busy-polled (no
// interrupt or idle-wake path). A SEND may carry two SGEs so a per-peer 128-B
// header can precede a body shared by all peers. Connection setup (QPN, PSN, GID,
// MTU) is exchanged by the caller over its TCP socket.

#include <infiniband/verbs.h>

#include <errno.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

typedef struct m26r_info {
    uint32_t qpn;
    uint32_t psn;
    uint8_t gid[16];
    uint32_t mtu;  // enum ibv_mtu
    uint32_t reserved;
} m26r_info;

typedef struct m26r_ep {
    struct ibv_context* ctx;
    struct ibv_pd* pd;
    struct ibv_cq* scq;
    struct ibv_cq* rcq;
    struct ibv_qp* qp;
    struct ibv_mr* send_mr;
    struct ibv_mr* recv_mr;
    struct ibv_mr* hdr_mr;
    uint8_t* send_buf;
    size_t send_len;
    uint8_t* recv_buf;
    size_t recv_len;
    uint32_t recv_slots;
    size_t slot_len;
    uint8_t* hdr_buf;
    size_t hdr_len;
    uint8_t port;
    int gid_index;
    union ibv_gid gid;
    uint32_t psn;
    enum ibv_mtu mtu;
} m26r_ep;

static void set_err(char* err, size_t len, const char* what, int e) {
    if (err && len) snprintf(err, len, "%s: %s (%d)", what, e ? strerror(e) : "failed", e);
}

static int64_t now_us(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (int64_t)ts.tv_sec * 1000000 + ts.tv_nsec / 1000;
}

void m26r_close(m26r_ep* ep) {
    if (!ep) return;
    if (ep->qp) ibv_destroy_qp(ep->qp);
    if (ep->scq) ibv_destroy_cq(ep->scq);
    if (ep->rcq) ibv_destroy_cq(ep->rcq);
    if (ep->send_mr) ibv_dereg_mr(ep->send_mr);
    if (ep->recv_mr) ibv_dereg_mr(ep->recv_mr);
    if (ep->hdr_mr) ibv_dereg_mr(ep->hdr_mr);
    if (ep->pd) ibv_dealloc_pd(ep->pd);
    if (ep->ctx) ibv_close_device(ep->ctx);
    free(ep);
}

// Open `dev` port `port` with GID `gid_index`, register the three caller-owned
// buffers (any may be NULL/0 except recv), create the RC QP and move it to INIT.
int m26r_open(const char* dev, uint8_t port, int gid_index, uint8_t* send_buf, size_t send_len,
              uint8_t* recv_buf, size_t recv_len, uint32_t recv_slots, uint8_t* hdr_buf, size_t hdr_len,
              m26r_ep** out, char* err, size_t errlen) {
    if (!dev || !out || !recv_buf || !recv_len || !recv_slots || recv_len % recv_slots) {
        set_err(err, errlen, "m26r_open: bad arguments", EINVAL);
        return -1;
    }
    m26r_ep* ep = calloc(1, sizeof(*ep));
    if (!ep) { set_err(err, errlen, "m26r_open: calloc", ENOMEM); return -1; }
    int n = 0;
    struct ibv_device** list = ibv_get_device_list(&n);
    if (!list) { set_err(err, errlen, "ibv_get_device_list", errno); free(ep); return -1; }
    for (int i = 0; i < n; ++i) {
        if (strcmp(ibv_get_device_name(list[i]), dev) == 0) { ep->ctx = ibv_open_device(list[i]); break; }
    }
    ibv_free_device_list(list);
    if (!ep->ctx) { set_err(err, errlen, "ibv_open_device (name not found?)", errno); free(ep); return -1; }
    struct ibv_port_attr pattr;
    if (ibv_query_port(ep->ctx, port, &pattr)) { set_err(err, errlen, "ibv_query_port", errno); m26r_close(ep); return -1; }
    if (pattr.state != IBV_PORT_ACTIVE) { set_err(err, errlen, "port not active", ENETDOWN); m26r_close(ep); return -1; }
    ep->mtu = pattr.active_mtu;
    ep->port = port;
    ep->gid_index = gid_index;
    if (ibv_query_gid(ep->ctx, port, gid_index, &ep->gid)) { set_err(err, errlen, "ibv_query_gid", errno); m26r_close(ep); return -1; }
    if (!(ep->pd = ibv_alloc_pd(ep->ctx))) { set_err(err, errlen, "ibv_alloc_pd", errno); m26r_close(ep); return -1; }
    ep->recv_buf = recv_buf; ep->recv_len = recv_len; ep->recv_slots = recv_slots; ep->slot_len = recv_len / recv_slots;
    if (!(ep->recv_mr = ibv_reg_mr(ep->pd, recv_buf, recv_len, IBV_ACCESS_LOCAL_WRITE))) {
        set_err(err, errlen, "ibv_reg_mr recv", errno); m26r_close(ep); return -1;
    }
    if (send_buf && send_len) {
        ep->send_buf = send_buf; ep->send_len = send_len;
        if (!(ep->send_mr = ibv_reg_mr(ep->pd, send_buf, send_len, IBV_ACCESS_LOCAL_WRITE))) {
            set_err(err, errlen, "ibv_reg_mr send", errno); m26r_close(ep); return -1;
        }
    }
    if (hdr_buf && hdr_len) {
        ep->hdr_buf = hdr_buf; ep->hdr_len = hdr_len;
        if (!(ep->hdr_mr = ibv_reg_mr(ep->pd, hdr_buf, hdr_len, IBV_ACCESS_LOCAL_WRITE))) {
            set_err(err, errlen, "ibv_reg_mr hdr", errno); m26r_close(ep); return -1;
        }
    }
    if (!(ep->scq = ibv_create_cq(ep->ctx, 64, NULL, NULL, 0)) ||
        !(ep->rcq = ibv_create_cq(ep->ctx, (int)recv_slots + 16, NULL, NULL, 0))) {
        set_err(err, errlen, "ibv_create_cq", errno); m26r_close(ep); return -1;
    }
    struct ibv_qp_init_attr qia;
    memset(&qia, 0, sizeof(qia));
    qia.send_cq = ep->scq;
    qia.recv_cq = ep->rcq;
    qia.qp_type = IBV_QPT_RC;
    qia.cap.max_send_wr = 32;
    qia.cap.max_recv_wr = recv_slots + 8;
    qia.cap.max_send_sge = 2;
    qia.cap.max_recv_sge = 1;
    if (!(ep->qp = ibv_create_qp(ep->pd, &qia))) { set_err(err, errlen, "ibv_create_qp", errno); m26r_close(ep); return -1; }
    struct ibv_qp_attr a;
    memset(&a, 0, sizeof(a));
    a.qp_state = IBV_QPS_INIT;
    a.pkey_index = 0;
    a.port_num = port;
    a.qp_access_flags = IBV_ACCESS_LOCAL_WRITE;
    int e = ibv_modify_qp(ep->qp, &a, IBV_QP_STATE | IBV_QP_PKEY_INDEX | IBV_QP_PORT | IBV_QP_ACCESS_FLAGS);
    if (e) { set_err(err, errlen, "ibv_modify_qp INIT", e); m26r_close(ep); return -1; }
    struct timespec ts;
    clock_gettime(CLOCK_REALTIME, &ts);
    ep->psn = (uint32_t)((ts.tv_nsec ^ (ts.tv_sec << 7) ^ (uintptr_t)ep) & 0xffffff);
    *out = ep;
    return 0;
}

void m26r_local_info(const m26r_ep* ep, m26r_info* out) {
    memset(out, 0, sizeof(*out));
    out->qpn = ep->qp->qp_num;
    out->psn = ep->psn;
    memcpy(out->gid, ep->gid.raw, 16);
    out->mtu = (uint32_t)ep->mtu;
}

// INIT -> RTR -> RTS against the remote peer (RoCE v2: global routing header).
int m26r_connect(m26r_ep* ep, const m26r_info* remote, char* err, size_t errlen) {
    struct ibv_qp_attr a;
    memset(&a, 0, sizeof(a));
    a.qp_state = IBV_QPS_RTR;
    a.path_mtu = (enum ibv_mtu)(remote->mtu < (uint32_t)ep->mtu ? remote->mtu : (uint32_t)ep->mtu);
    a.dest_qp_num = remote->qpn;
    a.rq_psn = remote->psn;
    a.max_dest_rd_atomic = 1;
    a.min_rnr_timer = 1;  // 0.01 ms: the peer re-posts its RECV right after each message
    a.ah_attr.is_global = 1;
    memcpy(a.ah_attr.grh.dgid.raw, remote->gid, 16);
    a.ah_attr.grh.sgid_index = (uint8_t)ep->gid_index;
    a.ah_attr.grh.hop_limit = 64;
    a.ah_attr.port_num = ep->port;
    int e = ibv_modify_qp(ep->qp, &a, IBV_QP_STATE | IBV_QP_AV | IBV_QP_PATH_MTU | IBV_QP_DEST_QPN | IBV_QP_RQ_PSN |
                                       IBV_QP_MAX_DEST_RD_ATOMIC | IBV_QP_MIN_RNR_TIMER);
    if (e) { set_err(err, errlen, "ibv_modify_qp RTR", e); return -1; }
    memset(&a, 0, sizeof(a));
    a.qp_state = IBV_QPS_RTS;
    a.sq_psn = ep->psn;
    // Local ACK timeout 4.096 us x 2^timeout. The inference fabric is lossy (no
    // PFC/ECN by policy): four Spark returns into the coordinator's one port overflow the
    // switch, and a lost tail packet waits out this timer. Measured on X1a 4K
    // (mimo26f-afd runs/20260925-reset/r2d): 14 (~67 ms, DS41RT's value) gave 40-150 ms
    // stalls; 10 (~4.2 ms) 2,988-3,465 tok/s; 8 (~1 ms) 3,580-3,655 tok/s.
    // GLM53F_RDMA_TIMEOUT overrides (1..31).
    a.timeout = 8;
    const char* t = getenv("GLM53F_RDMA_TIMEOUT");
    if (t && atoi(t) >= 1 && atoi(t) <= 31) a.timeout = (uint8_t)atoi(t);
    a.retry_cnt = 7;
    a.rnr_retry = 7;  // infinite RNR retry
    a.max_rd_atomic = 1;
    e = ibv_modify_qp(ep->qp, &a, IBV_QP_STATE | IBV_QP_TIMEOUT | IBV_QP_RETRY_CNT | IBV_QP_RNR_RETRY | IBV_QP_SQ_PSN |
                                   IBV_QP_MAX_QP_RD_ATOMIC);
    if (e) { set_err(err, errlen, "ibv_modify_qp RTS", e); return -1; }
    return 0;
}

int m26r_post_recv(m26r_ep* ep, uint32_t slot) {
    if (slot >= ep->recv_slots) return EINVAL;
    struct ibv_sge sge = {(uintptr_t)(ep->recv_buf + (size_t)slot * ep->slot_len), (uint32_t)ep->slot_len,
                          ep->recv_mr->lkey};
    struct ibv_recv_wr wr, *bad = NULL;
    memset(&wr, 0, sizeof(wr));
    wr.wr_id = slot;
    wr.sg_list = &sge;
    wr.num_sge = 1;
    return ibv_post_recv(ep->qp, &wr, &bad);
}

// SEND [hdr_buf + hdr_off, hdr_len] (skipped when hdr_len == 0) then
// [send_buf + body_off, body_len], signaled.
int m26r_post_send(m26r_ep* ep, size_t hdr_off, size_t hdr_len, size_t body_off, size_t body_len) {
    struct ibv_sge sge[2];
    int nsge = 0;
    if (hdr_len) {
        if (!ep->hdr_mr || hdr_off + hdr_len > ep->hdr_len) return EINVAL;
        sge[nsge].addr = (uintptr_t)(ep->hdr_buf + hdr_off);
        sge[nsge].length = (uint32_t)hdr_len;
        sge[nsge].lkey = ep->hdr_mr->lkey;
        ++nsge;
    }
    if (body_len) {
        if (!ep->send_mr || body_off + body_len > ep->send_len) return EINVAL;
        sge[nsge].addr = (uintptr_t)(ep->send_buf + body_off);
        sge[nsge].length = (uint32_t)body_len;
        sge[nsge].lkey = ep->send_mr->lkey;
        ++nsge;
    }
    if (!nsge) return EINVAL;
    struct ibv_send_wr wr, *bad = NULL;
    memset(&wr, 0, sizeof(wr));
    wr.wr_id = 0;
    wr.sg_list = sge;
    wr.num_sge = nsge;
    wr.opcode = IBV_WR_SEND;
    wr.send_flags = IBV_SEND_SIGNALED;
    return ibv_post_send(ep->qp, &wr, &bad);
}

// Poll one completion from `cq`. Spins for `spin_us`, then backs off with short
// sleeps until `timeout_us` (< 0: wait forever). Returns 0 with *wc filled, 1 on
// timeout, or -1 on a poll error.
static int poll_one(struct ibv_cq* cq, struct ibv_wc* wc, int64_t spin_us, int64_t timeout_us) {
    const int64_t t0 = now_us();
    for (;;) {
        const int n = ibv_poll_cq(cq, 1, wc);
        if (n < 0) return -1;
        if (n == 1) return 0;
        const int64_t dt = now_us() - t0;
        if (timeout_us >= 0 && dt >= timeout_us) return 1;
        if (dt >= spin_us) {
            struct timespec ts = {0, 20000};  // 20 us
            nanosleep(&ts, NULL);
        }
    }
}

// Wait for the SEND completion. Returns 0, 1 on timeout, or the negative WC status.
int m26r_wait_send(m26r_ep* ep, int64_t timeout_us) {
    struct ibv_wc wc;
    const int r = poll_one(ep->scq, &wc, 1000000, timeout_us);
    if (r) return r;
    return wc.status == IBV_WC_SUCCESS ? 0 : -(int)wc.status;
}

// Wait for one RECV completion: *slot, *len. Spins `spin_us` then backs off.
int m26r_wait_recv(m26r_ep* ep, int64_t spin_us, int64_t timeout_us, uint32_t* slot, uint32_t* len) {
    struct ibv_wc wc;
    const int r = poll_one(ep->rcq, &wc, spin_us, timeout_us);
    if (r) return r;
    if (wc.status != IBV_WC_SUCCESS) return -(int)wc.status;
    *slot = (uint32_t)wc.wr_id;
    *len = wc.byte_len;
    return 0;
}
