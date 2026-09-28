//! Routes: a request's top-8 table read into the kernel's inputs, and the
//! CPU twin of the device planner.
//!
//! A request carries 8 route entries per row (row index, expert id, FP32 gate
//! weight), token-major and slot-minor. [`request_routes`] and [`view_routes`]
//! check that shape (as mimo26f-afd's `serve::b1_inputs` and
//! `serve::serve_b1_view` do) and return the flat `ids` / `weights` arrays the
//! kernel takes.
//!
//! [`GroupPlan`] is what the device planner (`plan_kernel` in
//! `kernels/exl3_rank.cu`, after mimo26f-afd's one-CTA `plan_parallel`)
//! computes: every (row, slot) pair placed in ascending expert order, each
//! expert's pairs cut into groups of at most `group_rows`. The device places
//! an expert's pairs in atomic order, this twin in route order; both are valid
//! because every pair is computed on its own rows. Its tests pin the group
//! bound the launcher sizes its grid with.

use glm53f_wire::frame::{RequestFrame, RequestView};

use crate::consts::{EXPERTS, TOPK};

/// The flat `ids` and `weights` [rows * 8] of a decoded request.
pub fn request_routes(request: &RequestFrame) -> Result<(Vec<i32>, Vec<f32>), String> {
    let rows = request.rows.len();
    if request.hidden_rows.len() != rows {
        return Err(format!("routes: {} hidden rows for {rows} rows", request.hidden_rows.len()));
    }
    let mut ids = Vec::with_capacity(rows * TOPK);
    let mut weights = Vec::with_capacity(rows * TOPK);
    for (i, row) in request.rows.iter().enumerate() {
        if row.route_count as usize != TOPK {
            return Err(format!("routes: row {i} has {} routes, want {TOPK}", row.route_count));
        }
        let off = row.route_offset as usize;
        let rs = request.routes.get(off..off + TOPK).ok_or("routes: route range out of bounds")?;
        for r in rs {
            if r.row_index as usize != i {
                return Err(format!("routes: route of row {} under row {i}", r.row_index));
            }
            ids.push(r.expert_id as i32);
            weights.push(r.gate_weight);
        }
    }
    Ok((ids, weights))
}

/// [`request_routes`] for a request validated in place (the zero-copy
/// receive path): the entries are read straight from the frame.
pub fn view_routes(view: &RequestView<'_>) -> Result<(Vec<i32>, Vec<f32>), String> {
    let rows = view.rows;
    let mut ids = Vec::with_capacity(rows * TOPK);
    let mut weights = Vec::with_capacity(rows * TOPK);
    for i in 0..rows {
        let row = view.row(i).map_err(|e| e.to_string())?;
        if row.route_count as usize != TOPK {
            return Err(format!("routes: row {i} has {} routes, want {TOPK}", row.route_count));
        }
        let off = row.route_offset as usize; // parse checked off + count <= routes
        for j in off..off + TOPK {
            let r = view.route(j).map_err(|e| e.to_string())?;
            if r.row_index as usize != i {
                return Err(format!("routes: route of row {} under row {i}", r.row_index));
            }
            ids.push(r.expert_id as i32);
            weights.push(r.gate_weight);
        }
    }
    Ok((ids, weights))
}

/// One group: up to `group_rows` pairs of one expert, contiguous in pair order.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Group {
    pub expert: usize,
    /// First pair.
    pub start: usize,
    /// Pairs in the group.
    pub count: usize,
}

/// The device planner's output, computed on the host.
#[derive(Debug, Clone)]
pub struct GroupPlan {
    pub groups: Vec<Group>,
    /// Per pair: its row.
    pub pair_row: Vec<usize>,
    /// Per pair: its route (`row * 8 + slot`).
    pub pair_route: Vec<usize>,
    /// Per route: its pair.
    pub inverse: Vec<usize>,
}

/// The largest number of groups `rows` rows can form: at most `routes /
/// group_rows` full groups plus one partial group per distinct expert, and
/// never more groups than pairs. The launcher sizes the expert grids with it.
pub fn max_groups(rows: usize, group_rows: usize) -> usize {
    let routes = rows * TOPK;
    routes.min(routes / group_rows + routes.min(EXPERTS))
}

impl GroupPlan {
    /// Plan `ids` [rows * 8] (already checked, `kernel::check_routes`).
    pub fn new(ids: &[i32], group_rows: usize) -> Self {
        assert!(group_rows > 0);
        let routes = ids.len();
        let mut count = vec![0usize; EXPERTS];
        for &e in ids {
            count[e as usize] += 1;
        }
        let mut base = vec![0usize; EXPERTS];
        let mut running = 0;
        for e in 0..EXPERTS {
            base[e] = running;
            running += count[e];
        }
        let mut cursor = base.clone();
        let (mut pair_row, mut pair_route, mut inverse) = (vec![0; routes], vec![0; routes], vec![0; routes]);
        for (r, &e) in ids.iter().enumerate() {
            let gp = cursor[e as usize];
            cursor[e as usize] += 1;
            pair_row[gp] = r / TOPK;
            pair_route[gp] = r;
            inverse[r] = gp;
        }
        let mut groups = Vec::new();
        for e in 0..EXPERTS {
            let mut k = 0;
            while k < count[e] {
                let n = group_rows.min(count[e] - k);
                groups.push(Group { expert: e, start: base[e] + k, count: n });
                k += n;
            }
        }
        Self { groups, pair_row, pair_route, inverse }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::testkit::routes;

    fn check(ids: &[i32], gr: usize) {
        let rows = ids.len() / TOPK;
        let p = GroupPlan::new(ids, gr);
        // A bijection between routes and pairs.
        for (r, &gp) in p.inverse.iter().enumerate() {
            assert_eq!(p.pair_route[gp], r);
            assert_eq!(p.pair_row[gp], r / TOPK);
        }
        // Groups cover the pairs once, in ascending expert order, one expert each.
        let mut next = 0;
        let mut last_expert = 0;
        for g in &p.groups {
            assert_eq!(g.start, next);
            assert!(g.count >= 1 && g.count <= gr);
            assert!(g.expert >= last_expert);
            last_expert = g.expert;
            for gp in g.start..g.start + g.count {
                assert_eq!(ids[p.pair_route[gp]] as usize, g.expert);
            }
            next += g.count;
        }
        assert_eq!(next, ids.len());
        assert!(p.groups.len() <= max_groups(rows, gr), "{} groups > bound {}", p.groups.len(), max_groups(rows, gr));
    }

    #[test]
    fn plans_are_bijections_within_the_bound() {
        for rows in [1usize, 2, 7, 8, 64, 65, 300, 4096] {
            for gr in [16usize, 32] {
                for pool in [0usize, 8, 9, 40] {
                    let (ids, _) = routes(0xA11C_E000 + (rows * 7 + gr + pool) as u64, rows, pool);
                    check(&ids, gr);
                }
            }
        }
    }

    #[test]
    fn the_bound_is_reached() {
        // Every route a distinct expert (1 row): 8 groups of one pair each.
        let ids: Vec<i32> = (0..8).collect();
        assert_eq!(GroupPlan::new(&ids, 16).groups.len(), max_groups(1, 16));
        // All rows on the same 8 experts: 8 * ceil(rows / 16) groups.
        let rows = 100;
        let ids: Vec<i32> = (0..rows * TOPK).map(|r| (r % TOPK) as i32).collect();
        assert_eq!(GroupPlan::new(&ids, 16).groups.len(), 8 * 7);
        // Skewed: 17 pairs on one expert need 2 groups of 16.
        let mut ids: Vec<i32> = Vec::new();
        for r in 0..17 {
            ids.push(0);
            for s in 1..TOPK {
                ids.push((r * 7 + s) as i32 % 287 + 1);
            }
        }
        let p = GroupPlan::new(&ids, 16);
        assert_eq!(p.groups.iter().filter(|g| g.expert == 0).count(), 2);
    }
}
