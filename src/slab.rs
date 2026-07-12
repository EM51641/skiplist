use crate::order::{OrderId, Side};
use rust_decimal::Decimal;

pub type NodeId = u32;

pub struct OrderNode {
    pub id: OrderId,
    pub side: Side,
    pub price: Decimal,
    pub qty: i32,
    pub remaining: i32,
    pub prev: Option<NodeId>,
    pub next: Option<NodeId>,
}

pub struct Slab {
    nodes: Vec<Option<OrderNode>>,
    free: Vec<NodeId>,
}

impl Slab {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
            free: Vec::new(),
        }
    }

    pub fn with_capacity(cap: usize) -> Self {
        Self {
            nodes: Vec::with_capacity(cap),
            free: Vec::new(),
        }
    }

    pub fn insert(&mut self, node: OrderNode) -> NodeId {
        if let Some(id) = self.free.pop() {
            self.nodes[id as usize] = Some(node);
            id
        } else {
            let id = self.nodes.len() as NodeId;
            self.nodes.push(Some(node));
            id
        }
    }

    pub fn remove(&mut self, id: NodeId) -> Option<OrderNode> {
        let slot = self.nodes.get_mut(id as usize)?;
        let node = slot.take()?;
        self.free.push(id);
        Some(node)
    }

    pub fn get(&self, id: NodeId) -> Option<&OrderNode> {
        self.nodes.get(id as usize)?.as_ref()
    }

    pub fn get_mut(&mut self, id: NodeId) -> Option<&mut OrderNode> {
        self.nodes.get_mut(id as usize)?.as_mut()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(id: OrderId) -> OrderNode {
        OrderNode {
            id,
            side: Side::Buy,
            price: Decimal::new(10000, 2),
            qty: 10,
            remaining: 10,
            prev: None,
            next: None,
        }
    }

    #[test]
    fn insert_hands_out_sequential_ids_and_get_reads_back() {
        let mut slab = Slab::new();
        let a = slab.insert(node(1));
        let b = slab.insert(node(2));

        assert_eq!(a, 0);
        assert_eq!(b, 1);
        assert_eq!(slab.get(a).unwrap().id, 1);
        assert_eq!(slab.get(b).unwrap().id, 2);
    }

    #[test]
    fn get_mut_allows_mutation() {
        let mut slab = Slab::new();
        let a = slab.insert(node(1));

        slab.get_mut(a).unwrap().remaining = 3;

        assert_eq!(slab.get(a).unwrap().remaining, 3);
    }

    #[test]
    fn remove_returns_the_node_and_clears_the_slot() {
        let mut slab = Slab::new();
        let a = slab.insert(node(7));

        assert_eq!(slab.remove(a).unwrap().id, 7);
        assert!(slab.get(a).is_none());
    }

    #[test]
    fn removing_an_empty_slot_returns_none() {
        let mut slab = Slab::new();
        let a = slab.insert(node(1));

        slab.nodes[a as usize] = None;

        assert!(slab.remove(a).is_none());
    }

    #[test]
    fn freed_slot_is_reused_by_the_next_insert() {
        let mut slab = Slab::new();
        let a = slab.insert(node(1));
        let _b = slab.insert(node(2));

        // Free slot `a` by hand so this test exercises only `insert`'s reuse
        // path, independent of `remove`.
        slab.nodes[a as usize] = None;
        slab.free.push(a);

        let c = slab.insert(node(3));

        assert_eq!(c, a);
        assert_eq!(slab.get(c).unwrap().id, 3);
    }

    #[test]
    fn get_out_of_range_is_none() {
        let slab = Slab::new();

        assert!(slab.get(0).is_none());
    }
}
