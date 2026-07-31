// Exact, portable C++17 CPU closest-triangle kernel.  It intentionally uses
// only the CPython and NumPy C APIs so the repository does not acquire a new
// runtime dependency merely to accelerate the ambiguous spatial-FD path.

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <Python.h>
#include <numpy/arrayobject.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <limits>
#include <memory>
#include <queue>
#include <stdexcept>
#include <vector>

namespace {

struct Vec3 {
    double x, y, z;
    Vec3 operator+(const Vec3& other) const { return {x + other.x, y + other.y, z + other.z}; }
    Vec3 operator-(const Vec3& other) const { return {x - other.x, y - other.y, z - other.z}; }
    Vec3 operator*(double value) const { return {x * value, y * value, z * value}; }
};

double dot(const Vec3& a, const Vec3& b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
Vec3 cross(const Vec3& a, const Vec3& b) {
    return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x};
}

struct Triangle {
    Vec3 a, b, c;
    Vec3 lo, hi, centroid;
    int64_t face;
};

struct Node {
    Vec3 lo, hi;
    int start = 0;
    int stop = 0;
    int left = -1;
    int right = -1;
    bool leaf() const { return left < 0 && right < 0; }
};

struct Closest {
    Vec3 point;
    std::array<double, 3> bary;
    double distance2;
};

Closest closest_on_triangle(const Vec3& p, const Triangle& t) {
    const Vec3 ab = t.b - t.a;
    const Vec3 ac = t.c - t.a;
    if (dot(cross(ab, ac), cross(ab, ac)) <= 1e-30) {
        const std::array<Vec3, 3> vertices{t.a, t.b, t.c};
        double best = std::numeric_limits<double>::infinity();
        int index = 0;
        for (int i = 0; i < 3; ++i) {
            const double value = dot(p - vertices[i], p - vertices[i]);
            if (value < best) { best = value; index = i; }
        }
        std::array<double, 3> bary{0.0, 0.0, 0.0};
        bary[index] = 1.0;
        return {vertices[index], bary, best};
    }
    const Vec3 ap = p - t.a;
    const double d1 = dot(ab, ap);
    const double d2 = dot(ac, ap);
    if (d1 <= 0.0 && d2 <= 0.0) return {t.a, {1.0, 0.0, 0.0}, dot(ap, ap)};
    const Vec3 bp = p - t.b;
    const double d3 = dot(ab, bp);
    const double d4 = dot(ac, bp);
    if (d3 >= 0.0 && d4 <= d3) return {t.b, {0.0, 1.0, 0.0}, dot(bp, bp)};
    const double vc = d1 * d4 - d3 * d2;
    if (vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0) {
        const double v = d1 / (d1 - d3);
        const Vec3 q = t.a + ab * v;
        return {q, {1.0 - v, v, 0.0}, dot(p - q, p - q)};
    }
    const Vec3 cp = p - t.c;
    const double d5 = dot(ab, cp);
    const double d6 = dot(ac, cp);
    if (d6 >= 0.0 && d5 <= d6) return {t.c, {0.0, 0.0, 1.0}, dot(cp, cp)};
    const double vb = d5 * d2 - d1 * d6;
    if (vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0) {
        const double w = d2 / (d2 - d6);
        const Vec3 q = t.a + ac * w;
        return {q, {1.0 - w, 0.0, w}, dot(p - q, p - q)};
    }
    const double va = d3 * d6 - d5 * d4;
    if (va <= 0.0 && (d4 - d3) >= 0.0 && (d5 - d6) >= 0.0) {
        const Vec3 bc = t.c - t.b;
        const double w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        const Vec3 q = t.b + bc * w;
        return {q, {0.0, 1.0 - w, w}, dot(p - q, p - q)};
    }
    const double denominator = 1.0 / (va + vb + vc);
    const double v = vb * denominator;
    const double w = vc * denominator;
    const Vec3 q = t.a + ab * v + ac * w;
    return {q, {1.0 - v - w, v, w}, dot(p - q, p - q)};
}

double lower_bound2(const Vec3& p, const Node& node) {
    const double dx = std::max(std::max(node.lo.x - p.x, 0.0), p.x - node.hi.x);
    const double dy = std::max(std::max(node.lo.y - p.y, 0.0), p.y - node.hi.y);
    const double dz = std::max(std::max(node.lo.z - p.z, 0.0), p.z - node.hi.z);
    return dx * dx + dy * dy + dz * dz;
}

struct Mesh {
    std::vector<Triangle> triangles;
    std::vector<int> order;
    std::vector<Node> nodes;
    int leaf_size;
    std::atomic<uint64_t> query_count{0};
    std::atomic<uint64_t> point_count{0};
    std::atomic<uint64_t> candidates{0};
    std::atomic<uint64_t> node_visits{0};

    int build(int start, int stop) {
        const int index = static_cast<int>(nodes.size());
        nodes.push_back(Node{});
        Vec3 lo{std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity()};
        Vec3 hi{-std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity(), -std::numeric_limits<double>::infinity()};
        for (int position = start; position < stop; ++position) {
            const Triangle& t = triangles[order[position]];
            lo.x = std::min(lo.x, t.lo.x); lo.y = std::min(lo.y, t.lo.y); lo.z = std::min(lo.z, t.lo.z);
            hi.x = std::max(hi.x, t.hi.x); hi.y = std::max(hi.y, t.hi.y); hi.z = std::max(hi.z, t.hi.z);
        }
        const int count = stop - start;
        nodes[index].lo = lo; nodes[index].hi = hi;
        if (count <= leaf_size) { nodes[index].start = start; nodes[index].stop = stop; return index; }
        const std::array<double, 3> extent{hi.x - lo.x, hi.y - lo.y, hi.z - lo.z};
        const int axis = extent[1] > extent[0] ? (extent[2] > extent[1] ? 2 : 1) : (extent[2] > extent[0] ? 2 : 0);
        std::stable_sort(order.begin() + start, order.begin() + stop, [&](int lhs, int rhs) {
            const Vec3& a = triangles[lhs].centroid;
            const Vec3& b = triangles[rhs].centroid;
            const double av = axis == 0 ? a.x : (axis == 1 ? a.y : a.z);
            const double bv = axis == 0 ? b.x : (axis == 1 ? b.y : b.z);
            return av == bv ? triangles[lhs].face < triangles[rhs].face : av < bv;
        });
        const int middle = start + count / 2;
        const int left = build(start, middle);
        const int right = build(middle, stop);
        nodes[index].left = left; nodes[index].right = right;
        return index;
    }

    Closest query_one(const Vec3& point, int64_t* face_out) {
        using QueueItem = std::pair<double, int>;
        std::priority_queue<QueueItem, std::vector<QueueItem>, std::greater<QueueItem>> queue;
        queue.push({lower_bound2(point, nodes[0]), 0});
        double best = std::numeric_limits<double>::infinity();
        int64_t best_face = -1;
        Closest result{{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, best};
        while (!queue.empty()) {
            const auto [bound, index] = queue.top(); queue.pop();
            if (bound > best) break;
            node_visits.fetch_add(1, std::memory_order_relaxed);
            const Node& node = nodes[index];
            if (node.leaf()) {
                candidates.fetch_add(static_cast<uint64_t>(node.stop - node.start), std::memory_order_relaxed);
                for (int position = node.start; position < node.stop; ++position) {
                    const Triangle& triangle = triangles[order[position]];
                    const Closest candidate = closest_on_triangle(point, triangle);
                    if (candidate.distance2 < best || (candidate.distance2 == best && (best_face < 0 || triangle.face < best_face))) {
                        best = candidate.distance2; best_face = triangle.face; result = candidate;
                    }
                }
            } else {
                queue.push({lower_bound2(point, nodes[node.left]), node.left});
                queue.push({lower_bound2(point, nodes[node.right]), node.right});
            }
        }
        *face_out = best_face;
        return result;
    }
};

typedef struct { PyObject_HEAD Mesh* mesh; } HandleObject;
static PyTypeObject HandleType = {PyVarObject_HEAD_INIT(nullptr, 0)};

// The winding handle deliberately keeps triangles in input order.  That is the
// same deterministic reduction order used by the Python reference, without
// allocating an N-by-triangle temporary for a batch of query points.
struct WindingMesh { std::vector<Triangle> triangles; std::atomic<uint64_t> query_count{0}; std::atomic<uint64_t> point_count{0}; };
typedef struct { PyObject_HEAD WindingMesh* mesh; } WindingHandleObject;
static PyTypeObject WindingHandleType = {PyVarObject_HEAD_INIT(nullptr, 0)};

bool require_array(PyObject* value, int type, int dimensions, const char* name, PyArrayObject** output) {
    if (!PyArray_Check(value)) { PyErr_Format(PyExc_TypeError, "%s must be a NumPy array", name); return false; }
    auto* array = reinterpret_cast<PyArrayObject*>(value);
    if (PyArray_TYPE(array) != type) { PyErr_Format(PyExc_TypeError, "%s must have the required dtype", name); return false; }
    if (PyArray_NDIM(array) != dimensions || PyArray_DIMS(array)[dimensions - 1] != 3) { PyErr_Format(PyExc_ValueError, "%s must have shape (N, 3)", name); return false; }
    if (!PyArray_IS_C_CONTIGUOUS(array)) { PyErr_Format(PyExc_ValueError, "%s must be C-contiguous", name); return false; }
    *output = array;
    return true;
}

bool require_finite(const double* values, npy_intp count, const char* name) {
    for (npy_intp index = 0; index < count; ++index) if (!std::isfinite(values[index])) { PyErr_Format(PyExc_ValueError, "%s must contain only finite values", name); return false; }
    return true;
}

PyObject* handle_new(PyTypeObject* type, PyObject*, PyObject*) {
    auto* self = reinterpret_cast<HandleObject*>(type->tp_alloc(type, 0));
    if (self != nullptr) self->mesh = nullptr;
    return reinterpret_cast<PyObject*>(self);
}

int handle_init(HandleObject* self, PyObject* args, PyObject* kwargs) {
    PyObject *vertices_obj = nullptr, *faces_obj = nullptr;
    int leaf_size = 32;
    static const char* names[] = {"vertices", "faces", "leaf_size", nullptr};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO|i", const_cast<char**>(names), &vertices_obj, &faces_obj, &leaf_size)) return -1;
    if (leaf_size <= 0) { PyErr_SetString(PyExc_ValueError, "leaf_size must be positive"); return -1; }
    PyArrayObject *vertices = nullptr, *faces = nullptr;
    if (!require_array(vertices_obj, NPY_DOUBLE, 2, "vertices", &vertices)) return -1;
    if (!require_array(faces_obj, NPY_INT64, 2, "faces", &faces)) return -1;
    if (!require_finite(static_cast<const double*>(PyArray_DATA(vertices)), PyArray_SIZE(vertices), "vertices")) return -1;
    const npy_intp vertex_count = PyArray_DIMS(vertices)[0];
    const npy_intp face_count = PyArray_DIMS(faces)[0];
    if (vertex_count == 0 || face_count == 0) { PyErr_SetString(PyExc_ValueError, "compiled BVH requires non-empty vertices and faces"); return -1; }
    const auto* face_data = static_cast<const int64_t*>(PyArray_DATA(faces));
    for (npy_intp index = 0; index < face_count * 3; ++index) if (face_data[index] < 0 || face_data[index] >= vertex_count) { PyErr_SetString(PyExc_ValueError, "faces contain an invalid vertex index"); return -1; }
    const auto* vertex_data = static_cast<const double*>(PyArray_DATA(vertices));
    try {
        auto mesh = std::make_unique<Mesh>();
        mesh->leaf_size = leaf_size;
        mesh->triangles.reserve(static_cast<size_t>(face_count));
        for (npy_intp face = 0; face < face_count; ++face) {
            const auto load = [&](int64_t id) { return Vec3{vertex_data[id * 3], vertex_data[id * 3 + 1], vertex_data[id * 3 + 2]}; };
            Triangle t{load(face_data[face * 3]), load(face_data[face * 3 + 1]), load(face_data[face * 3 + 2]), {}, {}, {}, static_cast<int64_t>(face)};
            t.lo = {std::min({t.a.x, t.b.x, t.c.x}), std::min({t.a.y, t.b.y, t.c.y}), std::min({t.a.z, t.b.z, t.c.z})};
            t.hi = {std::max({t.a.x, t.b.x, t.c.x}), std::max({t.a.y, t.b.y, t.c.y}), std::max({t.a.z, t.b.z, t.c.z})};
            t.centroid = (t.lo + t.hi) * 0.5;
            mesh->triangles.push_back(t);
        }
        mesh->order.resize(mesh->triangles.size());
        for (size_t index = 0; index < mesh->order.size(); ++index) mesh->order[index] = static_cast<int>(index);
        mesh->build(0, static_cast<int>(mesh->order.size()));
        delete self->mesh; self->mesh = mesh.release();
    } catch (const std::exception& error) { PyErr_SetString(PyExc_RuntimeError, error.what()); return -1; }
    return 0;
}

void handle_dealloc(HandleObject* self) { delete self->mesh; Py_TYPE(self)->tp_free(reinterpret_cast<PyObject*>(self)); }

PyObject* handle_query(HandleObject* self, PyObject* args) {
    PyObject* points_obj = nullptr;
    if (!PyArg_ParseTuple(args, "O", &points_obj)) return nullptr;
    if (self->mesh == nullptr) { PyErr_SetString(PyExc_RuntimeError, "compiled BVH handle is destroyed or uninitialized"); return nullptr; }
    PyArrayObject* points = nullptr;
    if (!require_array(points_obj, NPY_DOUBLE, 2, "points_object_float64", &points)) return nullptr;
    if (!require_finite(static_cast<const double*>(PyArray_DATA(points)), PyArray_SIZE(points), "points_object_float64")) return nullptr;
    const npy_intp count = PyArray_DIMS(points)[0];
    npy_intp vector_dims[2] = {count, 3};
    PyArrayObject* closest = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(2, vector_dims, NPY_DOUBLE));
    PyArrayObject* faces = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(1, &count, NPY_INT64));
    PyArrayObject* bary = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(2, vector_dims, NPY_DOUBLE));
    PyArrayObject* distance = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(1, &count, NPY_DOUBLE));
    if (closest == nullptr || faces == nullptr || bary == nullptr || distance == nullptr) { Py_XDECREF(closest); Py_XDECREF(faces); Py_XDECREF(bary); Py_XDECREF(distance); return nullptr; }
    const auto* input = static_cast<const double*>(PyArray_DATA(points));
    auto* closest_data = static_cast<double*>(PyArray_DATA(closest)); auto* face_data = static_cast<int64_t*>(PyArray_DATA(faces));
    auto* bary_data = static_cast<double*>(PyArray_DATA(bary)); auto* distance_data = static_cast<double*>(PyArray_DATA(distance));
    self->mesh->query_count.fetch_add(1, std::memory_order_relaxed); self->mesh->point_count.fetch_add(static_cast<uint64_t>(count), std::memory_order_relaxed);
    Py_BEGIN_ALLOW_THREADS
    for (npy_intp index = 0; index < count; ++index) {
        int64_t face = -1; const Closest value = self->mesh->query_one({input[index * 3], input[index * 3 + 1], input[index * 3 + 2]}, &face);
        closest_data[index * 3] = value.point.x; closest_data[index * 3 + 1] = value.point.y; closest_data[index * 3 + 2] = value.point.z;
        bary_data[index * 3] = value.bary[0]; bary_data[index * 3 + 1] = value.bary[1]; bary_data[index * 3 + 2] = value.bary[2];
        face_data[index] = face; distance_data[index] = std::sqrt(std::max(value.distance2, 0.0));
    }
    Py_END_ALLOW_THREADS
    return Py_BuildValue("NNNN", closest, faces, bary, distance);
}

PyObject* handle_stats(HandleObject* self, PyObject*) {
    if (self->mesh == nullptr) { PyErr_SetString(PyExc_RuntimeError, "compiled BVH handle is destroyed or uninitialized"); return nullptr; }
    PyObject* result = PyDict_New();
    if (result == nullptr) return nullptr;
    const auto put = [&](const char* key, uint64_t value) { PyObject* item = PyLong_FromUnsignedLongLong(value); PyDict_SetItemString(result, key, item); Py_DECREF(item); };
    put("triangle_count", self->mesh->triangles.size()); put("node_count", self->mesh->nodes.size()); put("leaf_size", self->mesh->leaf_size);
    put("query_count", self->mesh->query_count.load()); put("queried_point_count", self->mesh->point_count.load()); put("candidate_triangle_evaluations", self->mesh->candidates.load()); put("node_visits", self->mesh->node_visits.load());
    return result;
}

PyMethodDef handle_methods[] = {{"query", reinterpret_cast<PyCFunction>(handle_query), METH_VARARGS, "Exact batch closest-triangle query."}, {"stats", reinterpret_cast<PyCFunction>(handle_stats), METH_NOARGS, "Return monotonic query statistics."}, {nullptr, nullptr, 0, nullptr}};
PyObject* winding_handle_new(PyTypeObject* type, PyObject*, PyObject*) {
    auto* self = reinterpret_cast<WindingHandleObject*>(type->tp_alloc(type, 0));
    if (self != nullptr) self->mesh = nullptr;
    return reinterpret_cast<PyObject*>(self);
}

int winding_handle_init(WindingHandleObject* self, PyObject* args, PyObject* kwargs) {
    PyObject *vertices_obj = nullptr, *faces_obj = nullptr;
    static const char* names[] = {"vertices", "faces", nullptr};
    if (!PyArg_ParseTupleAndKeywords(args, kwargs, "OO", const_cast<char**>(names), &vertices_obj, &faces_obj)) return -1;
    PyArrayObject *vertices = nullptr, *faces = nullptr;
    if (!require_array(vertices_obj, NPY_DOUBLE, 2, "vertices", &vertices)) return -1;
    if (!require_array(faces_obj, NPY_INT64, 2, "faces", &faces)) return -1;
    if (!require_finite(static_cast<const double*>(PyArray_DATA(vertices)), PyArray_SIZE(vertices), "vertices")) return -1;
    const npy_intp vertex_count = PyArray_DIMS(vertices)[0], face_count = PyArray_DIMS(faces)[0];
    if (vertex_count == 0 || face_count == 0) { PyErr_SetString(PyExc_ValueError, "compiled winding requires non-empty vertices and faces"); return -1; }
    const auto* face_data = static_cast<const int64_t*>(PyArray_DATA(faces));
    for (npy_intp index = 0; index < face_count * 3; ++index) if (face_data[index] < 0 || face_data[index] >= vertex_count) { PyErr_SetString(PyExc_ValueError, "faces contain an invalid vertex index"); return -1; }
    const auto* vertex_data = static_cast<const double*>(PyArray_DATA(vertices));
    try {
        auto mesh = std::make_unique<WindingMesh>(); mesh->triangles.reserve(static_cast<size_t>(face_count));
        for (npy_intp face = 0; face < face_count; ++face) {
            const auto load = [&](int64_t id) { return Vec3{vertex_data[id * 3], vertex_data[id * 3 + 1], vertex_data[id * 3 + 2]}; };
            mesh->triangles.push_back(Triangle{load(face_data[face * 3]), load(face_data[face * 3 + 1]), load(face_data[face * 3 + 2]), {}, {}, {}, static_cast<int64_t>(face)});
        }
        delete self->mesh; self->mesh = mesh.release();
    } catch (const std::exception& error) { PyErr_SetString(PyExc_RuntimeError, error.what()); return -1; }
    return 0;
}

void winding_handle_dealloc(WindingHandleObject* self) { delete self->mesh; Py_TYPE(self)->tp_free(reinterpret_cast<PyObject*>(self)); }

PyObject* winding_handle_query(WindingHandleObject* self, PyObject* args) {
    PyObject* points_obj = nullptr;
    if (!PyArg_ParseTuple(args, "O", &points_obj)) return nullptr;
    if (self->mesh == nullptr) { PyErr_SetString(PyExc_RuntimeError, "compiled winding handle is destroyed or uninitialized"); return nullptr; }
    PyArrayObject* points = nullptr;
    if (!require_array(points_obj, NPY_DOUBLE, 2, "points_object_float64", &points)) return nullptr;
    if (!require_finite(static_cast<const double*>(PyArray_DATA(points)), PyArray_SIZE(points), "points_object_float64")) return nullptr;
    const npy_intp count = PyArray_DIMS(points)[0];
    PyArrayObject* output = reinterpret_cast<PyArrayObject*>(PyArray_SimpleNew(1, &count, NPY_DOUBLE));
    if (output == nullptr) return nullptr;
    const auto* input = static_cast<const double*>(PyArray_DATA(points)); auto* values = static_cast<double*>(PyArray_DATA(output));
    self->mesh->query_count.fetch_add(1, std::memory_order_relaxed); self->mesh->point_count.fetch_add(static_cast<uint64_t>(count), std::memory_order_relaxed);
    constexpr double four_pi = 12.566370614359172953850573533118;
    Py_BEGIN_ALLOW_THREADS
    for (npy_intp index = 0; index < count; ++index) {
        const Vec3 p{input[index * 3], input[index * 3 + 1], input[index * 3 + 2]}; double total = 0.0;
        for (const Triangle& triangle : self->mesh->triangles) {
            const Vec3 a = triangle.a - p, b = triangle.b - p, c = triangle.c - p;
            const double la = std::sqrt(dot(a, a)), lb = std::sqrt(dot(b, b)), lc = std::sqrt(dot(c, c));
            const double numerator = dot(a, cross(b, c));
            const double denominator = la * lb * lc + dot(a, b) * lc + dot(b, c) * la + dot(c, a) * lb;
            total += 2.0 * std::atan2(numerator, denominator);
        }
        values[index] = total / four_pi;
    }
    Py_END_ALLOW_THREADS
    return reinterpret_cast<PyObject*>(output);
}

PyObject* winding_handle_stats(WindingHandleObject* self, PyObject*) {
    if (self->mesh == nullptr) { PyErr_SetString(PyExc_RuntimeError, "compiled winding handle is destroyed or uninitialized"); return nullptr; }
    PyObject* result = PyDict_New(); if (result == nullptr) return nullptr;
    const auto put = [&](const char* key, uint64_t value) { PyObject* item = PyLong_FromUnsignedLongLong(value); PyDict_SetItemString(result, key, item); Py_DECREF(item); };
    put("triangle_count", self->mesh->triangles.size()); put("query_count", self->mesh->query_count.load()); put("queried_point_count", self->mesh->point_count.load()); return result;
}

PyMethodDef winding_handle_methods[] = {{"query", reinterpret_cast<PyCFunction>(winding_handle_query), METH_VARARGS, "Exact batched generalized-winding query."}, {"stats", reinterpret_cast<PyCFunction>(winding_handle_stats), METH_NOARGS, "Return monotonic query statistics."}, {nullptr, nullptr, 0, nullptr}};
PyModuleDef module = {PyModuleDef_HEAD_INIT, "_compiled_sdf_cpu", "Portable exact CPU BVH kernel.", -1, nullptr, nullptr, nullptr, nullptr, nullptr};

}  // namespace

PyMODINIT_FUNC PyInit__compiled_sdf_cpu(void) {
    import_array();
    HandleType.tp_name = "_compiled_sdf_cpu.CompiledBVHHandle";
    HandleType.tp_basicsize = sizeof(HandleObject);
    HandleType.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    HandleType.tp_new = handle_new; HandleType.tp_init = reinterpret_cast<initproc>(handle_init); HandleType.tp_dealloc = reinterpret_cast<destructor>(handle_dealloc); HandleType.tp_methods = handle_methods;
    if (PyType_Ready(&HandleType) < 0) return nullptr;
    WindingHandleType.tp_name = "_compiled_sdf_cpu.CompiledGeneralizedWindingHandle";
    WindingHandleType.tp_basicsize = sizeof(WindingHandleObject);
    WindingHandleType.tp_flags = Py_TPFLAGS_DEFAULT | Py_TPFLAGS_BASETYPE;
    WindingHandleType.tp_new = winding_handle_new; WindingHandleType.tp_init = reinterpret_cast<initproc>(winding_handle_init); WindingHandleType.tp_dealloc = reinterpret_cast<destructor>(winding_handle_dealloc); WindingHandleType.tp_methods = winding_handle_methods;
    if (PyType_Ready(&WindingHandleType) < 0) return nullptr;
    PyObject* result = PyModule_Create(&module);
    if (result == nullptr) return nullptr;
    Py_INCREF(&HandleType);
    if (PyModule_AddObject(result, "CompiledBVHHandle", reinterpret_cast<PyObject*>(&HandleType)) < 0) { Py_DECREF(&HandleType); Py_DECREF(result); return nullptr; }
    Py_INCREF(&WindingHandleType);
    if (PyModule_AddObject(result, "CompiledGeneralizedWindingHandle", reinterpret_cast<PyObject*>(&WindingHandleType)) < 0) { Py_DECREF(&WindingHandleType); Py_DECREF(result); return nullptr; }
    return result;
}
