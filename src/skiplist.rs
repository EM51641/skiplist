use std::marker::PhantomData;
use std::ptr;
use rand;

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
            length: 0
        }
    }

    fn random_level(&mut self) -> usize {
        let mut lvl = 1;
        while lvl < self.max_level && rand::random::<f64>()  < 0.5 {
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

    #[test]
    fn insert_and_get() {
        let mut sl = SkipList::new(16);
        sl.insert(3, "three");
        sl.insert(1, "one");
        sl.insert(2, "two");

        assert_eq!(sl.get(&1), Some(&"one"));
        assert_eq!(sl.get(&2), Some(&"two"));
        assert_eq!(sl.get(&3), Some(&"three"));
        assert_eq!(sl.get(&4), None);
        assert_eq!(sl.len(), 3);
    }

    #[test]
    fn insert_replaces() {
        let mut sl = SkipList::new(16);
        assert_eq!(sl.insert(1, "a"), None);
        assert_eq!(sl.insert(1, "b"), Some("a"));
        assert_eq!(sl.get(&1), Some(&"b"));
        assert_eq!(sl.len(), 1);
    }

    #[test]
    fn remove() {
        let mut sl = SkipList::new(16);
        sl.insert(1, "one");
        sl.insert(2, "two");
        sl.insert(3, "three");

        assert_eq!(sl.remove(&2), Some("two"));
        assert_eq!(sl.get(&2), None);
        assert_eq!(sl.len(), 2);

        assert_eq!(sl.remove(&2), None);
        assert_eq!(sl.len(), 2);
    }

    #[test]
    fn iteration_is_sorted() {
        let mut sl = SkipList::new(16);
        for &v in &[5, 3, 8, 1, 4, 7, 2, 6] {
            sl.insert(v, v * 10);
        }
        let keys: Vec<_> = sl.iter().map(|(k, _)| *k).collect();
        assert_eq!(keys, vec![1, 2, 3, 4, 5, 6, 7, 8]);
    }

    #[test]
    fn range_query() {
        let mut sl = SkipList::new(16);
        for i in 0..10 {
            sl.insert(i, i * 100);
        }
        let pairs: Vec<_> = sl.range(&3, &7).map(|(k, v)| (*k, *v)).collect();
        assert_eq!(pairs, vec![(3, 300), (4, 400), (5, 500), (6, 600)]);
    }

    #[test]
    fn first_and_last() {
        let mut sl = SkipList::new(16);
        assert!(sl.first().is_none());
        assert!(sl.last().is_none());

        sl.insert(5, 'e');
        sl.insert(2, 'b');
        sl.insert(8, 'h');

        assert_eq!(sl.first(), Some((&2, &'b')));
        assert_eq!(sl.last(), Some((&8, &'h')));
    }

    #[test]
    fn stress() {
        let mut sl = SkipList::new(20);
        for i in 0..1000 {
            sl.insert(i, i);
        }
        assert_eq!(sl.len(), 1000);
        for i in 0..1000 {
            assert_eq!(sl.get(&i), Some(&i));
        }
        for i in (0..1000).step_by(2) {
            sl.remove(&i);
        }
        assert_eq!(sl.len(), 500);
        let keys: Vec<_> = sl.iter().map(|(k, _)| *k).collect();
        assert_eq!(keys, (0..1000).skip(1).step_by(2).collect::<Vec<_>>());
    }
}
