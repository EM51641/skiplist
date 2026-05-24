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
    pub fn new(side: Side) -> Self {
        Self {
            side,
            slab: Slab::new(),
            order_index: HashMap::new(),
            levels: SkipList::new(16),
        }
    }

    /// Map a real price to the skiplist key.
    /// Asks sort ascending (best = lowest). Bids invert so that
    /// best bid (highest real price) becomes the smallest key — the
    /// same skiplist gives you "best price first" for both sides.
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
