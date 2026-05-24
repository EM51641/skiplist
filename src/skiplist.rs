use rand;
use std::marker::PhantomData;
use std::ptr;

struct Node<K, V> {
    key: Option<K>,
    value: Option<V>,
    forward: Vec<*mut Node<K, V>>,
}

impl<K, V> Node<K, V> {
    fn new_sentinel(max_level: usize) -> *mut Self {
        Box::into_raw(Box::new(Node {
            key: None,
            value: None,
            forward: vec![ptr::null_mut(); max_level],
        }))
    }

    fn new(key: K, value: V, level: usize) -> *mut Self {
        Box::into_raw(Box::new(Node {
            key: Some(key),
            value: Some(value),
            forward: vec![ptr::null_mut(); level],
        }))
    }
}

unsafe fn get_fwd<K, V>(node: *mut Node<K, V>, level: usize) -> *mut Node<K, V> {
    unsafe { (&(*node).forward)[level] }
}

unsafe fn set_fwd<K, V>(node: *mut Node<K, V>, level: usize, target: *mut Node<K, V>) {
    unsafe {
        (&mut (*node).forward)[level] = target;
    }
}

/// A probabilistic sorted data structure providing O(log n) average-case
/// insert, search, and delete operations.
pub struct SkipList<K: Ord, V> {
    head: *mut Node<K, V>,
    max_level: usize,
    level: usize,
    length: usize,
}

// SAFETY: SkipList owns all its nodes exclusively. Mutation requires &mut self,
// so &SkipList can be safely shared (Sync) and moved (Send) across threads.
unsafe impl<K: Ord + Send, V: Send> Send for SkipList<K, V> {}
unsafe impl<K: Ord + Send + Sync, V: Send + Sync> Sync for SkipList<K, V> {}

impl<K: Ord, V> SkipList<K, V> {
    pub fn new(max_level: usize) -> Self {
        let max_level = max_level.max(1);
        SkipList {
            head: Node::new_sentinel(max_level),
            max_level,
            level: 0,
            length: 0,
        }
    }

    fn random_level(&mut self) -> usize {
        let mut lvl = 1;
        while lvl < self.max_level && rand::random::<f64>() < 0.5 {
            lvl += 1;
        }
        lvl
    }

    unsafe fn find(&self, key: &K) -> (Vec<*mut Node<K, V>>, *mut Node<K, V>) {
        unsafe {
            let mut update = vec![self.head; self.max_level];
            let mut current = self.head;

            for i in (0..self.level).rev() {
                loop {
                    let next = get_fwd(current, i);
                    if next.is_null() {
                        break;
                    }
                    if (*next).key.as_ref().unwrap() < key {
                        current = next;
                    } else {
                        break;
                    }
                }
                update[i] = current;
            }

            let candidate = if self.level > 0 {
                get_fwd(current, 0)
            } else {
                ptr::null_mut()
            };
            (update, candidate)
        }
    }

    pub fn insert(&mut self, key: K, value: V) -> Option<V> {
        unsafe {
            let (update, candidate) = self.find(&key);

            if !candidate.is_null() && (*candidate).key.as_ref().unwrap() == &key {
                let old = (*candidate).value.take();
                (*candidate).value = Some(value);
                return old;
            }

            let new_level = self.random_level();

            if new_level > self.level {
                self.level = new_level;
            }

            let new_node = Node::new(key, value, new_level);

            for i in 0..new_level {
                set_fwd(new_node, i, get_fwd(update[i], i));
                set_fwd(update[i], i, new_node);
            }

            self.length += 1;
            None
        }
    }

    pub fn get(&self, key: &K) -> Option<&V> {
        unsafe {
            let (_, candidate) = self.find(key);
            if !candidate.is_null() && (*candidate).key.as_ref().unwrap() == key {
                (*candidate).value.as_ref()
            } else {
                None
            }
        }
    }

    /// Mutably look up a value by key.
    pub fn get_mut(&mut self, key: &K) -> Option<&mut V> {
        unsafe {
            let (_, candidate) = self.find(key);
            if !candidate.is_null() && (*candidate).key.as_ref().unwrap() == key {
                (*candidate).value.as_mut()
            } else {
                None
            }
        }
    }

    /// Remove a key and return its value.
    pub fn remove(&mut self, key: &K) -> Option<V> {
        unsafe {
            let (update, candidate) = self.find(key);

            if candidate.is_null() || (*candidate).key.as_ref().unwrap() != key {
                return None;
            }

            for i in 0..self.level {
                if get_fwd(update[i], i) != candidate {
                    break;
                }
                set_fwd(update[i], i, get_fwd(candidate, i));
            }

            let value = (*candidate).value.take();
            drop(Box::from_raw(candidate));

            while self.level > 0 && get_fwd(self.head, self.level - 1).is_null() {
                self.level -= 1;
            }

            self.length -= 1;
            value
        }
    }

    pub fn contains(&self, key: &K) -> bool {
        self.get(key).is_some()
    }

    pub fn len(&self) -> usize {
        self.length
    }

    pub fn is_empty(&self) -> bool {
        self.length == 0
    }

    /// Smallest key-value pair (O(1)).
    pub fn first(&self) -> Option<(&K, &V)> {
        unsafe {
            let first = get_fwd(self.head, 0);
            if first.is_null() {
                None
            } else {
                Some((
                    (*first).key.as_ref().unwrap(),
                    (*first).value.as_ref().unwrap(),
                ))
            }
        }
    }

    /// Largest key-value pair (O(log n) average via top-down descent).
    pub fn last(&self) -> Option<(&K, &V)> {
        unsafe {
            let mut current = self.head;
            for i in (0..self.level).rev() {
                loop {
                    let next = get_fwd(current, i);
                    if next.is_null() {
                        break;
                    }
                    current = next;
                }
            }
            if current == self.head {
                None
            } else {
                Some((
                    (*current).key.as_ref().unwrap(),
                    (*current).value.as_ref().unwrap(),
                ))
            }
        }
    }

    /// Iterate all entries in ascending key order.
    pub fn iter<'a>(&'a self) -> Iter<'a, K, V> {
        unsafe {
            Iter {
                current: get_fwd(self.head, 0),
                _marker: PhantomData,
            }
        }
    }

    /// Iterate entries with keys in `[from, to)`.
    pub fn range<'a>(&'a self, from: &K, to: &'a K) -> RangeIter<'a, K, V> {
        unsafe {
            let mut current = self.head;

            for i in (0..self.level).rev() {
                loop {
                    let next = get_fwd(current, i);
                    if next.is_null() {
                        break;
                    }
                    if (*next).key.as_ref().unwrap() < from {
                        current = next;
                    } else {
                        break;
                    }
                }
            }

            RangeIter {
                current: get_fwd(current, 0),
                upper: to,
                _marker: PhantomData,
            }
        }
    }
}

impl<K: Ord, V> Drop for SkipList<K, V> {
    fn drop(&mut self) {
        unsafe {
            let mut current = get_fwd(self.head, 0);
            drop(Box::from_raw(self.head));
            while !current.is_null() {
                let next = get_fwd(current, 0);
                drop(Box::from_raw(current));
                current = next;
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Iterators
// ---------------------------------------------------------------------------

pub struct Iter<'a, K, V> {
    current: *mut Node<K, V>,
    _marker: PhantomData<&'a (K, V)>,
}

impl<'a, K: Ord, V> Iterator for Iter<'a, K, V> {
    type Item = (&'a K, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        if self.current.is_null() {
            return None;
        }
        unsafe {
            let node = &*self.current;
            self.current = (&node.forward)[0];
            Some((node.key.as_ref().unwrap(), node.value.as_ref().unwrap()))
        }
    }
}

pub struct RangeIter<'a, K, V> {
    current: *mut Node<K, V>,
    upper: &'a K,
    _marker: PhantomData<&'a (K, V)>,
}

impl<'a, K: Ord, V> Iterator for RangeIter<'a, K, V> {
    type Item = (&'a K, &'a V);

    fn next(&mut self) -> Option<Self::Item> {
        if self.current.is_null() {
            return None;
        }
        unsafe {
            let node = &*self.current;
            let key = node.key.as_ref().unwrap();
            if key >= self.upper {
                return None;
            }
            self.current = (&node.forward)[0];
            Some((key, node.value.as_ref().unwrap()))
        }
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use rstest::{fixture, rstest};
    use rust_decimal::Decimal;
    use rust_decimal_macros::dec;
    use std::collections::BTreeMap;

    const ENTRIES: &[(Decimal, i32)] = &[
        (dec!(5.0), 50),
        (dec!(3.5), 30),
        (dec!(8.25), 80),
        (dec!(1.1), 10),
        (dec!(4.0), 40),
        (dec!(7.75), 70),
        (dec!(2.5), 20),
        (dec!(6.0), 60),
        (dec!(9.9), 90),
        (dec!(0.0), 0),
    ];

    #[fixture]
    fn skiplist() -> SkipList<Decimal, i32> {
        let mut sl = SkipList::new(16);
        for (k, v) in ENTRIES {
            sl.insert(*k, *v);
        }
        sl
    }

    #[fixture]
    fn btreemap() -> BTreeMap<Decimal, i32> {
        ENTRIES.iter().copied().collect()
    }

    #[rstest]
    fn get_matches_btreemap(skiplist: SkipList<Decimal, i32>, btreemap: BTreeMap<Decimal, i32>) {
        let probes = [
            dec!(-1.0),
            dec!(0.0),
            dec!(1.1),
            dec!(2.5),
            dec!(3.5),
            dec!(4.0),
            dec!(5.0),
            dec!(6.0),
            dec!(7.75),
            dec!(8.25),
            dec!(9.9),
            dec!(10.0),
            dec!(3.499),
        ];
        for k in probes {
            assert_eq!(skiplist.get(&k), btreemap.get(&k), "mismatch at key {k}");
        }
        assert_eq!(skiplist.len(), btreemap.len());
    }

    #[rstest]
    fn delete_key(mut skiplist: SkipList<Decimal, i32>, mut btreemap: BTreeMap<Decimal, i32>) {
        let random_key = ENTRIES[rand::random_range(0..ENTRIES.len())].0;

        skiplist.remove(&random_key);
        btreemap.remove(&random_key);

        for (k, _) in ENTRIES {
            assert_eq!(skiplist.get(k), btreemap.get(k), "mismatch at key {k}");
        }
        assert_eq!(skiplist.len(), btreemap.len());
    }

    #[rstest]
    fn get_first(skiplist: SkipList<Decimal, i32>, btreemap: BTreeMap<Decimal, i32>) {
        let (sl_key, sl_val) = skiplist.first().unwrap();
        let (bt_key, bt_val) = btreemap.first_key_value().unwrap();
        assert_eq!(sl_key, bt_key);
        assert_eq!(sl_val, bt_val);
    }

    #[rstest]
    fn get_last(skiplist: SkipList<Decimal, i32>, btreemap: BTreeMap<Decimal, i32>) {
        let (sl_key, sl_val) = skiplist.last().unwrap();
        let (bt_key, bt_val) = btreemap.last_key_value().unwrap();

        assert_eq!(sl_key, bt_key);
        assert_eq!(sl_val, bt_val);
    }

    #[rstest]
    fn contains(skiplist: SkipList<Decimal, i32>) {
        assert_eq!(skiplist.contains(&dec!(5.0)), true);
        assert_eq!(skiplist.contains(&dec!(-1.0)), false);
    }

    #[rstest]
    fn empty(skiplist: SkipList<Decimal, i32>) {
        let newskiplist: SkipList<Decimal, i32> = SkipList::new(1);
        assert_eq!(skiplist.is_empty(), false);
        assert_eq!(newskiplist.is_empty(), true);
    }

    #[rstest]
    fn get_mut(mut skiplist: SkipList<Decimal, i32>) {
        let ptn = skiplist.get_mut(&dec!(1.1)).unwrap();
        *ptn = 100;
        assert_eq!(skiplist.get(&dec!(1.1)).unwrap(), &100);
    }

    #[rstest]
    #[case(dec!(3.0), dec!(7.0))]
    #[case(dec!(0.0), dec!(10.0))]
    #[case(dec!(-5.0), dec!(5.0))]
    #[case(dec!(5.0), dec!(100.0))]
    #[case(dec!(2.5), dec!(2.5))]
    #[case(dec!(50.0), dec!(100.0))]
    fn range_matches_btreemap(
        skiplist: SkipList<Decimal, i32>,
        btreemap: BTreeMap<Decimal, i32>,
        #[case] from: Decimal,
        #[case] to: Decimal,
    ) {
        let sl_range: Vec<_> = skiplist.range(&from, &to).map(|(k, v)| (*k, *v)).collect();
        let bt_range: Vec<_> = btreemap.range(from..to).map(|(k, v)| (*k, *v)).collect();
        assert_eq!(sl_range, bt_range);
    }
}
