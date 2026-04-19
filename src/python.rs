use std::cmp::Ordering;

use pyo3::exceptions::PyKeyError;
use pyo3::prelude::*;
use pyo3::types::PyList;

use crate::skiplist::SkipList;

/// Wrapper that gives f64 a total ordering (using `total_cmp`).
#[derive(Debug, Clone, Copy, PartialEq)]
struct OrdF64(f64);

impl Eq for OrdF64 {}

impl PartialOrd for OrdF64 {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for OrdF64 {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0.total_cmp(&other.0)
    }
}

/// A skip list mapping float keys to arbitrary Python values.
///
/// Maintains keys in sorted order with O(log n) average-case insert,
/// lookup, and deletion. Supports dict-like access, sorted iteration,
/// and range queries.
#[pyclass(name = "SkipList")]
struct PySkipList {
    inner: SkipList<OrdF64, PyObject>,
}

#[pymethods]
impl PySkipList {
    #[new]
    #[pyo3(signature = (max_level = 16))]
    fn new(max_level: usize) -> Self {
        PySkipList {
            inner: SkipList::new(max_level),
        }
    }

    /// Insert a key-value pair. Returns the previous value if the key
    /// existed, otherwise None.
    fn insert(&mut self, key: f64, value: PyObject) -> Option<PyObject> {
        self.inner.insert(OrdF64(key), value)
    }

    /// Get the value for `key`, or `default` if not present.
    #[pyo3(signature = (key, default = None))]
    fn get(&self, py: Python<'_>, key: f64, default: Option<PyObject>) -> Option<PyObject> {
        self.inner
            .get(&OrdF64(key))
            .map(|v| v.clone_ref(py))
            .or(default)
    }

    /// Remove a key and return its value. Raises KeyError if missing.
    fn remove(&mut self, key: f64) -> PyResult<PyObject> {
        self.inner
            .remove(&OrdF64(key))
            .ok_or_else(|| PyKeyError::new_err(key))
    }

    /// Pop a key: return its value or `default`. Raises KeyError if
    /// missing and no default given.
    #[pyo3(signature = (key, default = None))]
    fn pop(&mut self, key: f64, default: Option<PyObject>) -> PyResult<PyObject> {
        match self.inner.remove(&OrdF64(key)) {
            Some(v) => Ok(v),
            None => default.ok_or_else(|| PyKeyError::new_err(key)),
        }
    }

    /// Check membership.
    fn contains(&self, key: f64) -> bool {
        self.inner.contains(&OrdF64(key))
    }

    /// Smallest key-value pair, or None.
    fn first(&self, py: Python<'_>) -> Option<PyObject> {
        self.inner
            .first()
            .map(|(k, v)| (k.0, v.clone_ref(py)).into_pyobject(py).unwrap().into())
    }

    /// Largest key-value pair, or None.
    fn last(&self, py: Python<'_>) -> Option<PyObject> {
        self.inner
            .last()
            .map(|(k, v)| (k.0, v.clone_ref(py)).into_pyobject(py).unwrap().into())
    }

    /// Return all (key, value) pairs with key in [lo, hi), sorted.
    fn range(&self, py: Python<'_>, lo: f64, hi: f64) -> PyResult<Py<PyList>> {
        let items: Vec<PyObject> = self
            .inner
            .range(&OrdF64(lo), &OrdF64(hi))
            .map(|(k, v)| (k.0, v.clone_ref(py)).into_pyobject(py).unwrap().into())
            .collect();
        Ok(PyList::new(py, items)?.into())
    }

    /// All keys in sorted order.
    fn keys(&self, py: Python<'_>) -> PyResult<Py<PyList>> {
        let keys: Vec<f64> = self.inner.iter().map(|(k, _)| k.0).collect();
        Ok(PyList::new(py, keys)?.into())
    }

    /// All values in key-sorted order.
    fn values(&self, py: Python<'_>) -> PyResult<Py<PyList>> {
        let vals: Vec<PyObject> = self
            .inner
            .iter()
            .map(|(_, v)| v.clone_ref(py))
            .collect();
        Ok(PyList::new(py, vals)?.into())
    }

    /// All (key, value) pairs in sorted order.
    fn items(&self, py: Python<'_>) -> PyResult<Py<PyList>> {
        let items: Vec<PyObject> = self
            .inner
            .iter()
            .map(|(k, v)| (k.0, v.clone_ref(py)).into_pyobject(py).unwrap().into())
            .collect();
        Ok(PyList::new(py, items)?.into())
    }

    // -- dunder methods --

    fn __len__(&self) -> usize {
        self.inner.len()
    }

    fn __bool__(&self) -> bool {
        !self.inner.is_empty()
    }

    fn __contains__(&self, key: f64) -> bool {
        self.inner.contains(&OrdF64(key))
    }

    fn __getitem__(&self, py: Python<'_>, key: f64) -> PyResult<PyObject> {
        self.inner
            .get(&OrdF64(key))
            .map(|v| v.clone_ref(py))
            .ok_or_else(|| PyKeyError::new_err(key))
    }

    fn __setitem__(&mut self, key: f64, value: PyObject) {
        self.inner.insert(OrdF64(key), value);
    }

    fn __delitem__(&mut self, key: f64) -> PyResult<()> {
        self.inner
            .remove(&OrdF64(key))
            .map(|_| ())
            .ok_or_else(|| PyKeyError::new_err(key))
    }

    fn __repr__(&self) -> String {
        let preview: Vec<String> = self
            .inner
            .iter()
            .take(5)
            .map(|(k, _)| format!("{}", k.0))
            .collect();
        let suffix = if self.inner.len() > 5 { ", ..." } else { "" };
        format!(
            "SkipList(len={}, keys=[{}{}])",
            self.inner.len(),
            preview.join(", "),
            suffix,
        )
    }
}

#[pymodule]
pub fn matching_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySkipList>()?;
    Ok(())
}
