use crate::skiplist::SkipList;
use rust_decimal::Decimal;
use std::collections::HashMap;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Side {
    Buy,
    Sell,
}

pub type OrderId = u64;
pub type NodeId = u32;

pub struct NewOrder {
    pub id: OrderId,
    pub side: Side,
    pub price: Decimal,
    pub qty: i32,
}

pub struct CancelOrder {
    pub id: OrderId,
}

pub struct ModifyOrder {
    pub id: OrderId,
    pub new_qty: i32,
}

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

#[derive(Clone, Copy)]
pub struct Level {
    pub head: Option<NodeId>,
    pub tail: Option<NodeId>,
    pub total_qty: i32,
}

pub struct Book {
    side: Side,
    slab: Slab,
    order_index: HashMap<OrderId, NodeId>,
    levels: SkipList<Decimal, Level>,
}

impl Book {
    pub fn new(
        side: Side,
        levels: Option<SkipList<Decimal, Level>>,
        slab: Option<Slab>,
        order_index: Option<HashMap<OrderId, NodeId>>,
    ) -> Self {
        let levels = levels.unwrap_or_else(|| SkipList::new(16));
        let slab = slab.unwrap_or_else(|| Slab::new());
        let order_index = order_index.unwrap_or_else(|| HashMap::new());

        Self {
            side,
            slab,
            order_index,
            levels,
        }
    }

    fn key(&self, price: Decimal) -> Decimal {
        match self.side {
            Side::Sell => price,
            Side::Buy => -price,
        }
    }

    pub fn submit_order(&mut self, req: NewOrder) {
        let nid = self.slab.insert(OrderNode {
            id: req.id,
            side: req.side,
            price: req.price,
            qty: req.qty,
            remaining: req.qty,
            prev: None,
            next: None,
        });

        let k = self.key(req.price);
        match self.levels.get_mut(&k) {
            Some(level) => {
                let prev_tail = level.tail;
                level.tail = Some(nid);
                level.total_qty += req.qty;
                if level.head.is_none() {
                    level.head = Some(nid);
                }
                if let Some(t) = prev_tail {
                    self.slab.get_mut(t).unwrap().next = Some(nid);
                    self.slab.get_mut(nid).unwrap().prev = Some(t);
                }
            }
            None => {
                self.levels.insert(
                    k,
                    Level {
                        head: Some(nid),
                        tail: Some(nid),
                        total_qty: req.qty,
                    },
                );
            }
        }

        self.order_index.insert(req.id, nid);
    }

    pub fn cancel_order(&mut self, req: CancelOrder) -> Option<OrderNode> {
        let nid = self.order_index.remove(&req.id)?;
        let (price, remaining, prev, next) = {
            let n = self.slab.get(nid)?;
            (n.price, n.remaining, n.prev, n.next)
        };

        if let Some(p) = prev {
            self.slab.get_mut(p).unwrap().next = next;
        }
        if let Some(n) = next {
            self.slab.get_mut(n).unwrap().prev = prev;
        }

        let k = self.key(price);
        let level_empty = match self.levels.get_mut(&k) {
            Some(level) => {
                if level.head == Some(nid) {
                    level.head = next;
                }
                if level.tail == Some(nid) {
                    level.tail = prev;
                }
                level.total_qty -= remaining;
                level.head.is_none()
            }
            None => false,
        };
        if level_empty {
            self.levels.remove(&k);
        }

        self.slab.remove(nid)
    }

    pub fn modify_order(&mut self, req: ModifyOrder) -> Option<()> {
        let nid = *self.order_index.get(&req.id)?;
        let (price, old_remaining) = {
            let n = self.slab.get(nid)?;
            (n.price, n.remaining)
        };
        self.slab.get_mut(nid)?.remaining = req.new_qty;
        let k = self.key(price);
        if let Some(level) = self.levels.get_mut(&k) {
            level.total_qty += req.new_qty - old_remaining;
        }
        Some(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rust_decimal::Decimal;

    fn px(cents: i64) -> Decimal {
        Decimal::new(cents, 2)
    }

    mod submit {
        use super::*;

        #[test]
        fn into_empty_level_creates_isolated_node() {
            let side = Side::Sell;
            let price = px(10000);

            let mut book = Book::new(side, None, None, None);
            book.submit_order(NewOrder {
                id: 1,
                side,
                price,
                qty: 10,
            });

            let level = book.levels.get(&price).unwrap();
            assert_eq!(level.head, level.tail);
            assert!(level.head.is_some());
            assert_eq!(level.total_qty, 10);
        }

        #[test]
        fn appends_to_existing_level() {
            let side = Side::Sell;
            let price = px(10000);

            let mut slab = Slab::new();
            let nid_a = slab.insert(OrderNode {
                id: 1,
                side,
                price,
                qty: 10,
                remaining: 10,
                prev: None,
                next: None,
            });
            let nid_b = slab.insert(OrderNode {
                id: 2,
                side,
                price,
                qty: 20,
                remaining: 20,
                prev: Some(nid_a),
                next: None,
            });
            slab.get_mut(nid_a).unwrap().next = Some(nid_b);

            let mut order_index = HashMap::new();
            order_index.insert(1, nid_a);
            order_index.insert(2, nid_b);

            let mut levels = SkipList::new(16);
            levels.insert(
                price,
                Level {
                    head: Some(nid_a),
                    tail: Some(nid_b),
                    total_qty: 30,
                },
            );

            let mut book = Book::new(side, Some(levels), Some(slab), Some(order_index));

            book.submit_order(NewOrder {
                id: 3,
                side,
                price,
                qty: 30,
            });

            let level = book.levels.get(&price).unwrap();
            assert_eq!(level.head, Some(nid_a));
            assert_ne!(level.tail, Some(nid_b), "tail must advance past prior tail");
            assert_eq!(level.total_qty, 60);
        }
    }

    mod cancel {
        use super::*;

        #[test]
        fn only_order_removes_level() {
            let side = Side::Sell;
            let price = px(10000);

            let mut slab = Slab::new();
            let nid = slab.insert(OrderNode {
                id: 1,
                side,
                price,
                qty: 10,
                remaining: 10,
                prev: None,
                next: None,
            });

            let mut order_index = HashMap::new();
            order_index.insert(1, nid);

            let mut levels = SkipList::new(16);
            levels.insert(
                price,
                Level {
                    head: Some(nid),
                    tail: Some(nid),
                    total_qty: 10,
                },
            );

            let mut book = Book::new(side, Some(levels), Some(slab), Some(order_index));

            let removed = book.cancel_order(CancelOrder { id: 1 });

            assert!(removed.is_some());
            assert_eq!(removed.unwrap().id, 1);
            assert!(book.order_index.is_empty());
            assert!(book.levels.get(&price).is_none());
        }

        #[test]
        fn head_relinks_and_updates_level() {
            let side = Side::Sell;
            let price = px(10000);

            let mut slab = Slab::new();
            let nid_a = slab.insert(OrderNode {
                id: 1,
                side,
                price,
                qty: 10,
                remaining: 10,
                prev: None,
                next: None,
            });
            let nid_b = slab.insert(OrderNode {
                id: 2,
                side,
                price,
                qty: 20,
                remaining: 20,
                prev: Some(nid_a),
                next: None,
            });
            let nid_c = slab.insert(OrderNode {
                id: 3,
                side,
                price,
                qty: 30,
                remaining: 30,
                prev: Some(nid_b),
                next: None,
            });
            slab.get_mut(nid_a).unwrap().next = Some(nid_b);
            slab.get_mut(nid_b).unwrap().next = Some(nid_c);

            let mut order_index = HashMap::new();
            order_index.insert(1, nid_a);
            order_index.insert(2, nid_b);
            order_index.insert(3, nid_c);

            let mut levels = SkipList::new(16);
            levels.insert(
                price,
                Level {
                    head: Some(nid_a),
                    tail: Some(nid_c),
                    total_qty: 60,
                },
            );

            let mut book = Book::new(side, Some(levels), Some(slab), Some(order_index));

            book.cancel_order(CancelOrder { id: 1 });

            let level = book.levels.get(&price).unwrap();
            assert_eq!(level.head, Some(nid_b));
            assert_eq!(level.tail, Some(nid_c));
            assert_eq!(level.total_qty, 50);
        }

        #[test]
        fn tail_relinks_and_updates_level() {
            let side = Side::Sell;
            let price = px(10000);

            let mut slab = Slab::new();
            let nid_a = slab.insert(OrderNode {
                id: 1,
                side,
                price,
                qty: 10,
                remaining: 10,
                prev: None,
                next: None,
            });
            let nid_b = slab.insert(OrderNode {
                id: 2,
                side,
                price,
                qty: 20,
                remaining: 20,
                prev: Some(nid_a),
                next: None,
            });
            let nid_c = slab.insert(OrderNode {
                id: 3,
                side,
                price,
                qty: 30,
                remaining: 30,
                prev: Some(nid_b),
                next: None,
            });
            slab.get_mut(nid_a).unwrap().next = Some(nid_b);
            slab.get_mut(nid_b).unwrap().next = Some(nid_c);

            let mut order_index = HashMap::new();
            order_index.insert(1, nid_a);
            order_index.insert(2, nid_b);
            order_index.insert(3, nid_c);

            let mut levels = SkipList::new(16);
            levels.insert(
                price,
                Level {
                    head: Some(nid_a),
                    tail: Some(nid_c),
                    total_qty: 60,
                },
            );

            let mut book = Book::new(side, Some(levels), Some(slab), Some(order_index));

            book.cancel_order(CancelOrder { id: 3 });

            let level = book.levels.get(&price).unwrap();
            assert_eq!(level.head, Some(nid_a));
            assert_eq!(level.tail, Some(nid_b));
            assert_eq!(level.total_qty, 30);
        }

        #[test]
        fn middle_relinks_neighbors_and_updates_level() {
            let side = Side::Sell;
            let price = px(10000);

            let mut slab = Slab::new();
            let nid_a = slab.insert(OrderNode {
                id: 1,
                side,
                price,
                qty: 10,
                remaining: 10,
                prev: None,
                next: None,
            });
            let nid_b = slab.insert(OrderNode {
                id: 2,
                side,
                price,
                qty: 20,
                remaining: 20,
                prev: Some(nid_a),
                next: None,
            });
            let nid_c = slab.insert(OrderNode {
                id: 3,
                side,
                price,
                qty: 30,
                remaining: 30,
                prev: Some(nid_b),
                next: None,
            });
            slab.get_mut(nid_a).unwrap().next = Some(nid_b);
            slab.get_mut(nid_b).unwrap().next = Some(nid_c);

            let mut order_index = HashMap::new();
            order_index.insert(1, nid_a);
            order_index.insert(2, nid_b);
            order_index.insert(3, nid_c);

            let mut levels = SkipList::new(16);
            levels.insert(
                price,
                Level {
                    head: Some(nid_a),
                    tail: Some(nid_c),
                    total_qty: 60,
                },
            );

            let mut book = Book::new(side, Some(levels), Some(slab), Some(order_index));

            book.cancel_order(CancelOrder { id: 2 });

            let level = book.levels.get(&price).unwrap();
            assert_eq!(level.head, Some(nid_a));
            assert_eq!(level.tail, Some(nid_c));
            assert_eq!(level.total_qty, 40);
        }
    }

    mod modify {
        use super::*;

        #[test]
        fn preserves_linkage_and_adjusts_level_total() {
            let side = Side::Sell;
            let price = px(10000);

            let mut slab = Slab::new();
            let nid_a = slab.insert(OrderNode {
                id: 1,
                side,
                price,
                qty: 10,
                remaining: 10,
                prev: None,
                next: None,
            });
            let nid_b = slab.insert(OrderNode {
                id: 2,
                side,
                price,
                qty: 20,
                remaining: 20,
                prev: Some(nid_a),
                next: None,
            });
            slab.get_mut(nid_a).unwrap().next = Some(nid_b);

            let mut order_index = HashMap::new();
            order_index.insert(1, nid_a);
            order_index.insert(2, nid_b);

            let mut levels = SkipList::new(16);
            levels.insert(
                price,
                Level {
                    head: Some(nid_a),
                    tail: Some(nid_b),
                    total_qty: 30,
                },
            );

            let mut book = Book::new(side, Some(levels), Some(slab), Some(order_index));

            assert!(
                book.modify_order(ModifyOrder {
                    id: 2,
                    new_qty: 50
                })
                .is_some()
            );

            let level = book.levels.get(&price).unwrap();
            assert_eq!(level.head, Some(nid_a));
            assert_eq!(level.tail, Some(nid_b));
            assert_eq!(level.total_qty, 60);
        }
    }
}
