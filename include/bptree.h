#pragma once

#include <cassert>
#include <functional>
#include <iostream>
#include <list>
#include <memory>
#include <utility>

namespace tree_details {

template <class T>
constexpr bool IsConst = false;
template <template <bool> class T>
constexpr bool IsConst<T<true>> = true;

} // namespace tree_details

template <class Key, class Value, std::size_t BlockSize = 4096, class Less = std::less<Key>>
class BPTree
{
    using key_type = Key;
    using mapped_type = Value;
    using value_type = std::pair<Key, Value>; // NB: a digression from std::map
    using reference = value_type &;
    using const_reference = const value_type &;
    using pointer = value_type *;
    using const_pointer = const value_type *;
    using size_type = std::size_t;

    struct Leaf;

    struct LessPair
    {
        bool operator()(const value_type & lhs, const Key & rhs)
        {
            return comp(lhs.first, rhs);
        }
    };

    struct Node
    {
        virtual std::pair<std::unique_ptr<Node>, Leaf *> clone(Leaf * sib_link) const = 0;

        virtual ~Node()
        {
        }
    };

    struct Leaf : Node
    {
        static constexpr size_type capacity = (BlockSize - sizeof(size_type) - sizeof(void *)) / (sizeof(value_type)) - 1;
        Leaf * r_sibling = nullptr;

        std::array<value_type, capacity + 1> storage;
        using iterator = typename std::array<value_type, capacity>::iterator;
        size_type size = 0;

        Leaf() = default;

        Leaf(const Leaf & other)
            : storage(other.storage)
            , size(other.size)
        {
        }

        std::pair<std::unique_ptr<Node>, Leaf *> clone(Leaf * sib_link) const override
        {
            return clone_impl(sib_link);
        }

        template <class K, class V>
        void push_back(K && k, V && v)
        {
            storage[size] = std::make_pair(std::forward<K>(k), std::forward<V>(v));
            size++;
        }

        template <class K, class V>
        void insert(K && k, V && v, size_type pos)
        {
            for (size_type i = size; i > pos; --i) {
                storage[i] = std::move(storage[i - 1]);
            }
            storage[pos] = {std::forward<K>(k), std::forward<V>(v)};
            size++;
        }

        void erase(size_type pos)
        {
            for (size_type i = pos + 1; i < size; ++i) {
                storage[i - 1] = std::move(storage[i]);
            }
            size--;
        }

        virtual ~Leaf()
        {
        }

    private:
        template <class R = Value, std::enable_if_t<std::is_copy_constructible_v<R>, int> = 0>
        std::pair<std::unique_ptr<Node>, Leaf *> clone_impl(Leaf * sib_link) const
        {
            std::unique_ptr<Leaf> copy = std::make_unique<Leaf>(*this);
            copy->r_sibling = sib_link;
            return {std::move(copy), copy.get()};
        }

        template <class R = Value, std::enable_if_t<!std::is_copy_constructible_v<R>, int> = 0>
        std::pair<std::unique_ptr<Node>, Leaf *> clone_impl(Leaf *) const
        {
            throw std::bad_function_call{};
        }
    };

    struct Internal : Node
    {
        static constexpr size_type capacity = (BlockSize - sizeof(size_type) - sizeof(void *)) / (sizeof(key_type) + sizeof(void *)) - 1;

        std::array<key_type, capacity + 1> keys;
        std::array<std::unique_ptr<Node>, capacity + 2> children;
        size_type size = 0;

        Internal() = default;

        std::pair<std::unique_ptr<Node>, Leaf *> clone(Leaf * sib_link) const override
        {
            std::unique_ptr<Internal> copy = std::make_unique<Internal>();
            copy->keys = keys;
            copy->size = size;
            for (size_type i = size + 1; i > 0; --i) {
                auto cp = children[i - 1]->clone(sib_link);
                sib_link = cp.second;
                copy->children[i - 1] = std::move(cp.first);
            }
            return {std::move(copy), sib_link};
        }
    };

    template <bool is_const>
    class Iterator
    {
    public:
        using difference_type = std::ptrdiff_t;
        using value_type = std::conditional_t<is_const, const BPTree::value_type, BPTree::value_type>;
        using leaf_type = std::conditional_t<is_const, const Leaf, Leaf>;
        using pointer = value_type *;
        using reference = value_type &;
        using iterator_category = std::forward_iterator_tag;

        Iterator() = default;

        template <class R = Iterator, std::enable_if_t<tree_details::IsConst<R>, int> = 0>
        Iterator(const Iterator<false> & other)
            : Iterator(other.m_current_data, other.m_current_node, *other.m_tree)
        {
        }

        Iterator(const Iterator &) = default;

        friend bool operator==(const Iterator & lhs, const Iterator & rhs)
        {
            return lhs.m_tree == rhs.m_tree && lhs.m_current_node == rhs.m_current_node && ((lhs.m_current_data == std::nullopt && rhs.m_current_data == std::nullopt) || (lhs.m_current_data != std::nullopt && rhs.m_current_data != std::nullopt && *lhs.m_current_data == *rhs.m_current_data));
        }
        friend bool operator!=(const Iterator & lhs, const Iterator & rhs)
        {
            return !(operator==(lhs, rhs));
        }

        reference operator*() const
        {
            return **m_current_data;
        }

        pointer operator->() const
        {
            return *m_current_data;
        }

        Iterator & operator++()
        {
            (*this->m_current_data)++;
            if (*this->m_current_data == this->m_current_node->storage.begin() + this->m_current_node->size) {
                if (this->m_current_node->r_sibling != nullptr) {
                    this->m_current_node = this->m_current_node->r_sibling;
                    this->m_current_data = this->m_current_node->storage.begin();
                }
                else {
                    this->m_current_node = nullptr;
                    this->m_current_data = std::nullopt;
                }
            }
            return *this;
        }

        Iterator operator++(int)
        {
            auto tmp = *this;
            operator++();
            return tmp;
        }

    private:
        friend class BPTree;

        Iterator(const BPTree & tree)
            : m_tree(&tree)
        {
        }

        using iterator_type = std::conditional_t<is_const, const std::pair<Key, Value> *, std::pair<Key, Value> *>;
        Iterator(iterator_type iter, leaf_type * node, const BPTree & tree)
            : m_tree(&tree)
            , m_current_node(node)
            , m_current_data(iter)
        {
        }

        Iterator(std::optional<iterator_type> iter, leaf_type * node, const BPTree & tree)
            : m_tree(&tree)
            , m_current_node(node)
            , m_current_data(iter)
        {
        }

        const BPTree * m_tree = nullptr;
        leaf_type * m_current_node = nullptr;
        std::optional<iterator_type> m_current_data = std::nullopt;
    };

public:
    using iterator = Iterator<false>;
    using const_iterator = Iterator<true>;

private:
    template <bool is_const>
    static std::conditional_t<is_const, const Leaf, Leaf> * find_leaf(std::vector<std::pair<std::conditional_t<is_const, const Node, Node> *, size_type>> &, const key_type &);

    void split(std::vector<std::pair<Node *, size_type>> &);

    template <bool Const, class K, class V>
    std::pair<Iterator<Const>, bool> insert(K &&, V &&);

    std::pair<iterator, bool> erase_impl(key_type);

    bool erase_from_leaf(Leaf *, const Key &);

    bool borrow(std::vector<std::pair<Node *, size_type>> &);

    bool borrow_parametric(Node *, Node *, size_type, size_type);

    void fix_int_keys(std::vector<std::pair<Node *, size_type>>);

    Node * merge(std::vector<std::pair<Node *, size_type>> &);

    Node * try_merge_parametric(Node *, Node *, Key);

    Leaf * left(const std::unique_ptr<Node> &);

    Leaf * left(Node *);

    template <bool is_const, class T>
    static Iterator<is_const> lower_bound(T *, const key_type &);

    template <bool is_const>
    Iterator<is_const> upper_bound(const key_type &);

    template <bool is_const, class T>
    static Iterator<is_const> find(T *, const key_type &);

    template <bool is_const, class T>
    static std::pair<Iterator<is_const>, Iterator<is_const>> equal_range(T *, const key_type &);

    template <bool is_const, class T>
    static std::conditional_t<is_const, const mapped_type, mapped_type> & at_impl(T *, const key_type &);

    std::unique_ptr<Node> m_root;
    size_type m_size = 0;

    static Less comp;
    static LessPair comp_pair;

public:
    BPTree()
        : m_root(std::make_unique<Leaf>())
    {
    }

    BPTree(std::initializer_list<value_type> && initializer_list)
        : m_root(std::make_unique<Leaf>())
    {
        insert(std::move(initializer_list));
    }

    BPTree(const BPTree & other)
        : m_root(other.m_root->clone(nullptr).first)
        , m_size(other.size())
    {
    }

    BPTree & operator=(const BPTree & other)
    {
        m_size = other.size();
        m_root = other.m_root->clone(nullptr).first;
        return *this;
    }

    BPTree(BPTree && other)
        : m_root(std::move(other.m_root))
        , m_size(other.m_size)
    {
    }

    BPTree & operator=(BPTree && other)
    {
        m_root = std::move(other.m_root);
        std::swap(m_size, other.m_size);
        return *this;
    }

    iterator begin()
    {
        if (empty())
            return end();
        Leaf * left_leaf = left(m_root);
        return {left_leaf->storage.begin(), left_leaf, *this};
    }
    const_iterator cbegin() const
    {
        if (empty())
            return cend();
        Leaf * left_leaf = left(m_root);
        return {left_leaf->storage.begin(), left_leaf, *this};
    }
    const_iterator begin() const
    {
        return cbegin();
    }
    iterator end()
    {
        return iterator(*this);
    }
    const_iterator cend() const
    {
        return const_iterator(*this);
    }
    const_iterator end() const
    {
        return cend();
    }

    bool empty() const
    {
        return m_size == 0;
    }

    size_type size() const
    {
        return m_size;
    }

    void clear()
    {
        m_root = std::make_unique<Leaf>();
        m_size = 0;
    }

    size_type count(const key_type & key) const
    {
        return contains(key) ? 1 : 0;
    }
    bool contains(const key_type & key) const
    {
        const_iterator it = find(key);
        return it != end() && !comp(it->first, key) && !comp(key, it->first);
    }
    std::pair<iterator, iterator> equal_range(const key_type & key)
    {
        return equal_range<false>(this, key);
    }
    std::pair<const_iterator, const_iterator> equal_range(const key_type & key) const
    {
        return equal_range<true>(this, key);
    }
    iterator lower_bound(const key_type & key)
    {
        return lower_bound<false>(this, key);
    }
    const_iterator lower_bound(const key_type & key) const
    {
        return lower_bound<true>(this, key);
    }
    iterator upper_bound(const key_type & key)
    {
        return upper_bound<false>(key);
    }
    const_iterator upper_bound(const key_type & key) const
    {
        return upper_bound<true>(key);
    }
    iterator find(const key_type & key)
    {
        return find<false>(this, key);
    }
    const_iterator find(const key_type & key) const
    {
        return find<true>(this, key);
    }

    // 'at' method throws std::out_of_range if there is no such key
    mapped_type & at(const key_type & key)
    {
        return at_impl<false>(this, key);
    }
    const mapped_type & at(const key_type & key) const
    {
        return at_impl<true>(this, key);
    }

    // '[]' operator inserts a new element if there is no such key
    mapped_type & operator[](const key_type & key)
    {
        auto it = find(key);
        if (it == end()) {
            return insert(key, mapped_type()).first->second;
        }
        return it->second;
    }

    std::pair<iterator, bool> insert(const key_type & key, const mapped_type & val)
    { // NB: a digression from std::map
        return insert<false>(key, val);
    }

    std::pair<iterator, bool> insert(const key_type & key, mapped_type && val) // NB: a digression from std::map
    {
        return insert<false>(key, std::move(val));
    }

    template <class ForwardIt>
    void insert(ForwardIt begin, ForwardIt end)
    {
        for (auto i = begin; i != end; i++) {
            insert(i->first, i->second);
        }
    }

    void insert(std::initializer_list<value_type> initializer_list)
    {
        for (value_type i : initializer_list) {
            insert(std::move(i.first), std::move(i.second));
        }
    }
    iterator erase(const_iterator it)
    {
        return erase_impl(it->first).first;
    }

    iterator erase(const_iterator _begin, const_iterator _end)
    {
        std::vector<Key> keys;
        iterator ret;
        for (const_iterator it = _begin; it != _end; ++it) {
            keys.push_back(it->first);
        }
        for (const auto & key : keys) {
            ret = erase_impl(key).first;
        }
        return ret;
    }

    size_type erase(const key_type & key)
    {
        auto ans = erase_impl(key).second ? 1 : 0;
        return ans;
    }
};

template <class Key, class Value, std::size_t BlockSize, class Less>
Less BPTree<Key, Value, BlockSize, Less>::comp;

template <class Key, class Value, std::size_t BlockSize, class Less>
typename BPTree<Key, Value, BlockSize, Less>::LessPair BPTree<Key, Value, BlockSize, Less>::comp_pair;

template <class Key, class Value, std::size_t BlockSize, class Less>
template <bool is_const, class T>
auto BPTree<Key, Value, BlockSize, Less>::find(T * self, const key_type & key)
        -> Iterator<is_const>
{
    Iterator<is_const> lb = self->lower_bound(key);
    if (lb != self->end() && !comp(lb->first, key) && !comp(key, lb->first)) {
        return lb;
    }
    return self->end();
}

template <class Key, class Value, std::size_t BlockSize, class Less>
template <bool is_const, class T>
auto BPTree<Key, Value, BlockSize, Less>::equal_range(T * self, const key_type & key)
        -> std::pair<Iterator<is_const>, Iterator<is_const>>
{
    Iterator<is_const> lb = lower_bound<is_const>(self, key);
    if (lb != self->end() && !comp(lb->first, key) && !comp(key, lb->first)) {
        Iterator<is_const> cp = lb;
        return {cp, ++lb};
    }
    return {lb, lb};
}

template <class Key, class Value, std::size_t BlockSize, class Less>
template <bool is_const, class T>
auto BPTree<Key, Value, BlockSize, Less>::at_impl(T * self, const key_type & key)
        -> std::conditional_t<is_const, const mapped_type, mapped_type> &
{
    auto it = self->find(key);
    if (it == self->end()) {
        throw std::out_of_range("element not found");
    }
    return it->second;
}

template <class Key, class Value, std::size_t BlockSize, class Less>
template <bool is_const>
auto BPTree<Key, Value, BlockSize, Less>::find_leaf(std::vector<std::pair<std::conditional_t<is_const, const Node, Node> *, size_type>> & path, const key_type & key)
        -> std::conditional_t<is_const, const Leaf, Leaf> *
{
    using internal_type = std::conditional_t<is_const, const Internal, Internal>;
    using leaf_type = std::conditional_t<is_const, const Leaf, Leaf>;
    const internal_type * cur_node = dynamic_cast<internal_type *>(path.back().first);

    if (cur_node == nullptr) {
        leaf_type * res = dynamic_cast<leaf_type *>(path.back().first);
        if (res == nullptr) {
            throw std::logic_error("unknown node type");
        }
        return res;
    }

    auto it = std::upper_bound(cur_node->keys.begin(), cur_node->keys.begin() + cur_node->size, key, comp);

    size_type pos = it - cur_node->keys.begin();
    path.emplace_back(cur_node->children[pos].get(), pos);
    return find_leaf<is_const>(path, key); // Tailing recursion
}

template <class Key, class Value, std::size_t BlockSize, class Less>
void BPTree<Key, Value, BlockSize, Less>::split(std::vector<std::pair<Node *, size_type>> & path)
{
    Node * origin_node = path.back().first;

    key_type new_delimiter;
    std::unique_ptr<Node> new_node;

    Leaf * origin_leaf = dynamic_cast<Leaf *>(origin_node);
    if (origin_leaf != nullptr) {
        if (origin_leaf->size <= Leaf::capacity) {
            return;
        }

        std::unique_ptr<Leaf> new_leaf = std::make_unique<Leaf>();
        new_leaf->r_sibling = origin_leaf->r_sibling;
        origin_leaf->r_sibling = new_leaf.get();

        for (size_type i = (Leaf::capacity + 1) / 2; i <= Leaf::capacity; ++i) {
            new_leaf->storage[i - (Leaf::capacity + 1) / 2] = std::move(origin_leaf->storage[i]);
            new_leaf->size++;
            origin_leaf->size--;
        }

        new_delimiter = new_leaf->storage[0].first;
        new_node = std::move(new_leaf);
    }
    else {
        Internal * origin_internal = dynamic_cast<Internal *>(origin_node);
        if (origin_internal == nullptr) {
            throw std::logic_error("Unknown node type");
        }

        if (origin_internal->size <= Internal::capacity) {
            return;
        }

        std::unique_ptr<Internal> new_internal = std::make_unique<Internal>();
        for (size_type i = (Internal::capacity + 1) / 2 + 1; i <= Internal::capacity; ++i) {
            new_internal->keys[i - (Internal::capacity + 1) / 2 - 1] = std::move(origin_internal->keys[i]);
            new_internal->size++;
            origin_internal->size--;
        }
        for (size_type i = (Internal::capacity + 1) / 2 + 1; i <= Internal::capacity + 1; ++i) {
            new_internal->children[i - (Internal::capacity + 1) / 2 - 1] = std::move(origin_internal->children[i]);
        }

        new_delimiter = std::move(origin_internal->keys[(Internal::capacity + 1) / 2]);
        origin_internal->size--;
        new_node = std::move(new_internal);
    }

    path.pop_back();
    if (path.empty()) {
        assert(m_root.get() == origin_node);
        std::unique_ptr<Internal> parent = std::make_unique<Internal>();

        parent->children[0] = std::move(m_root);
        m_root = std::move(parent);

        path.emplace_back(m_root.get(), 0);
    }

    Internal * parent = dynamic_cast<Internal *>(path.back().first);
    if (parent == nullptr) {
        throw std::logic_error("Parent is Leaf");
    }

    auto it = std::lower_bound(parent->keys.begin(), parent->keys.begin() + parent->size, new_delimiter, comp);
    size_type pos = it - parent->keys.begin();
    for (size_type i = parent->size; i > pos; --i) {
        parent->keys[i] = std::move(parent->keys[i - 1]);
    }
    for (size_type i = parent->size + 1; i > pos + 1; --i) {
        parent->children[i] = std::move(parent->children[i - 1]);
    }
    parent->keys[pos] = std::move(new_delimiter);
    parent->children[pos + 1] = std::move(new_node);
    parent->size++;
    split(path);
}

template <class Key, class Value, std::size_t BlockSize, class Less>
template <bool Const, class K, class V>
auto BPTree<Key, Value, BlockSize, Less>::insert(K && key, V && value)
        -> std::pair<Iterator<Const>, bool>
{
    std::vector<std::pair<Node *, size_type>> path;
    path.emplace_back(m_root.get(), 0);
    Leaf * target = find_leaf<false>(path, key);

    auto it = std::lower_bound(target->storage.begin(), target->storage.begin() + target->size, key, comp_pair);

    size_type pos = it - target->storage.begin();
    if (it == target->storage.begin() + target->size) {
        target->template push_back(std::forward<K>(key), std::forward<V>(value));
    }
    else {
        if (!comp(it->first, key) && !comp(key, it->first)) {
            return {Iterator<Const>(it, target, *this), false};
        }
        target->template insert(std::forward<K>(key), std::forward<V>(value), pos);
    }

    split(path);

    it = std::lower_bound(target->storage.begin(), target->storage.begin() + target->size, key, comp_pair);
    if (it == target->storage.begin() + target->size || comp(it->first, key) || comp(key, it->first)) {
        it = std::lower_bound(target->r_sibling->storage.begin(), target->r_sibling->storage.begin() + target->r_sibling->size, key, comp_pair);
        target = target->r_sibling;
    }

    m_size++;
    return {Iterator<Const>(it, target, *this), true};
}

template <class Key, class Value, std::size_t BlockSize, class Less>
auto BPTree<Key, Value, BlockSize, Less>::erase_impl(const key_type key)
        -> std::pair<iterator, bool>
{
    std::vector<std::pair<Node *, size_type>> path;
    path.emplace_back(m_root.get(), 0);
    Leaf * target = find_leaf<false>(path, key);

    if (!erase_from_leaf(target, key)) {
        return {{std::lower_bound(target->storage.begin(), target->storage.begin() + target->size, key, comp_pair), target, *this}, false};
    }

    if (target->size > 0) {
        fix_int_keys(path);
    }

    m_size--;

    if (target->size < (Leaf::capacity + 1) / 2) {
        if (!borrow(path)) {
            target = dynamic_cast<Leaf *>(merge(path));
            if (target == nullptr) {
                throw std::logic_error("Merge changed node type");
            }
        }
    }

    auto it = std::lower_bound(target->storage.begin(), target->storage.begin() + target->size, key, comp_pair);
    if (it == target->storage.begin() + target->size) {
        if (target->r_sibling != nullptr) {
            return {{target->r_sibling->storage.begin(), target->r_sibling, *this}, true};
        }
        return {end(), true};
    }
    return {{it, target, *this}, true};
}

template <class Key, class Value, std::size_t BlockSize, class Less>
bool BPTree<Key, Value, BlockSize, Less>::erase_from_leaf(Leaf * target, const Key & key)
{
    auto it = std::lower_bound(target->storage.begin(), target->storage.begin() + target->size, key, comp_pair);

    if (it == target->storage.begin() + target->size || comp(it->first, key) || comp(key, it->first)) {
        return false;
    }

    target->erase(it - target->storage.begin());

    return true;
}

template <class Key, class Value, std::size_t BlockSize, class Less>
bool BPTree<Key, Value, BlockSize, Less>::borrow(std::vector<std::pair<Node *, size_type>> & path)
{
    Node * cur = path.back().first;
    size_type pos = path.back().second;
    if (path.size() > 1) {
        Internal * par = dynamic_cast<Internal *>(path[path.size() - 2].first);
        if (par == nullptr) {
            throw std::logic_error("Parent is leaf");
        }

        if (pos > 0) {
            if (borrow_parametric(par->children[pos - 1].get(), cur, 0, 1)) { // non-zero pos is calculated automatically
                path.emplace_back(left(cur), 0);
                fix_int_keys(path);
                path.pop_back();
                Internal * cur_internal = dynamic_cast<Internal *>(cur);
                if (cur_internal != nullptr) {
                    path.emplace_back(left(cur_internal->children[1]), 1);
                    fix_int_keys(path);
                    path.pop_back();
                }
                return true;
            }
        }
        if (pos < par->size) {
            if (borrow_parametric(par->children[pos + 1].get(), cur, 1, 0)) { // non-zero pos is calculated automatically
                path.emplace_back(left(cur), 0);
                fix_int_keys(path);
                path.pop_back();
                Internal * cur_internal = dynamic_cast<Internal *>(cur);
                if (cur_internal != nullptr) {
                    path.emplace_back(left(cur_internal->children[cur_internal->size]), cur_internal->size);
                    fix_int_keys(path);
                    path.pop_back();
                }
                path.pop_back();
                path.emplace_back(par->children[pos + 1].get(), pos + 1);
                path.emplace_back(left(par->children[pos + 1]), 0);
                fix_int_keys(path);
                path.pop_back();
                return true;
            }
        }
    }
    return false;
}

template <class Key, class Value, std::size_t BlockSize, class Less>
bool BPTree<Key, Value, BlockSize, Less>::borrow_parametric(Node * from, Node * to, size_type insertion_pos, size_type extraction_pos)
{
    Leaf * cur_leaf = dynamic_cast<Leaf *>(to);
    Internal * cur_internal = dynamic_cast<Internal *>(to);

    Leaf * from_leaf = dynamic_cast<Leaf *>(from);

    if (cur_leaf != nullptr && from_leaf != nullptr && from_leaf->size > (Leaf::capacity + 1) / 2) {
        if (extraction_pos != 0) {
            extraction_pos = from_leaf->size - 1;
        }
        if (insertion_pos != 0) {
            insertion_pos = cur_leaf->size;
        }
        cur_leaf->template insert(std::move(from_leaf->storage[extraction_pos].first), std::move(from_leaf->storage[extraction_pos].second), insertion_pos);
        from_leaf->erase(extraction_pos);
        return true;
    }
    Internal * from_internal = dynamic_cast<Internal *>(from);
    if (cur_internal != nullptr && from_internal != nullptr && from_internal->size > (Internal::capacity + 1) / 2) {
        if (insertion_pos == 0) {
            for (size_type i = cur_internal->size; i > insertion_pos; --i) {
                cur_internal->keys[i] = std::move(cur_internal->keys[i - 1]);
            }
            for (size_type i = cur_internal->size + 1; i > insertion_pos; --i) {
                cur_internal->children[i] = std::move(cur_internal->children[i - 1]);
            }
            cur_internal->keys[insertion_pos] = std::move(from_internal->keys[from_internal->size - 1]);
            cur_internal->children[insertion_pos] = std::move(from_internal->children[from_internal->size]);
        }

        if (extraction_pos == 0) {
            cur_internal->keys[cur_internal->size] = std::move(from_internal->keys[extraction_pos]);
            cur_internal->children[cur_internal->size + 1] = std::move(from_internal->children[extraction_pos]);

            for (size_type i = extraction_pos; i < from_internal->size - 1; ++i) {
                cur_internal->keys[i] = std::move(cur_internal->keys[i + 1]);
            }
            for (size_type i = extraction_pos; i < from_internal->size; ++i) {
                cur_internal->children[i] = std::move(cur_internal->children[i + 1]);
            }
        }

        cur_internal->size++;
        from_internal->size--;
        return true;
    }
    return false;
}

template <class Key, class Value, std::size_t BlockSize, class Less>
void BPTree<Key, Value, BlockSize, Less>::fix_int_keys(const std::vector<std::pair<Node *, size_type>> path)
{
    Key key;
    Internal * cur_internal = dynamic_cast<Internal *>(path.back().first);
    if (cur_internal != nullptr) {
        key = cur_internal->keys[0];
    }
    else {
        Leaf * cur_leaf = dynamic_cast<Leaf *>(path.back().first);
        if (cur_leaf != nullptr) {
            key = cur_leaf->storage[0].first;
        }
        else {
            throw std::logic_error("Unknown node type");
        }
    }

    auto it = path.rbegin();
    while (it != path.rend() && it->second == 0) {
        it++;
    }
    if (it != path.rend()) {
        size_type pos = it->second - 1;
        Internal * target = dynamic_cast<Internal *>((++it)->first);
        if (target == nullptr) {
            throw std::logic_error("Parent is leaf");
        }
        target->keys[pos] = key;
    }
}

template <class Key, class Value, std::size_t BlockSize, class Less>
auto BPTree<Key, Value, BlockSize, Less>::merge(std::vector<std::pair<Node *, size_type>> & path)
        -> Node *
{
    if (path.size() <= 1) {
        return path.back().first;
    }

    Internal * par = dynamic_cast<Internal *>(path[path.size() - 2].first);
    if (par == nullptr) {
        throw std::logic_error("Parent is leaf");
    }
    size_type pos = path.back().second;
    Node * cur = path.back().first;

    Node * res = nullptr;

    if (pos > 0) {
        res = try_merge_parametric(par->children[pos - 1].get(), cur, par->keys[pos - 1]);
        if (res != nullptr) {
            path.pop_back();
            path.emplace_back(res, pos - 1);
            for (size_type i = pos - 1; i < par->size - 1; ++i) {
                par->keys[i] = std::move(par->keys[i + 1]);
            }
            for (size_type i = pos; i < par->size; ++i) {
                par->children[i] = std::move(par->children[i + 1]);
            }
            par->size--;
        }
    }
    if (res == nullptr && pos < par->size) {
        res = try_merge_parametric(cur, par->children[pos + 1].get(), par->keys[pos]);
        if (res != nullptr) {
            path.pop_back();
            path.emplace_back(res, pos);
            for (size_type i = pos; i < par->size - 1; ++i) {
                par->keys[i] = std::move(par->keys[i + 1]);
            }
            for (size_type i = pos + 1; i < par->size; ++i) {
                par->children[i] = std::move(par->children[i + 1]);
            }
            par->size--;
        }
    }
    if (res == nullptr) {
        return cur;
    }

    path.emplace_back(left(res), 0);
    fix_int_keys(path);
    path.pop_back();
    path.pop_back();

    if (par == m_root.get()) {
        if (par->size == 0) {
            Internal * root = dynamic_cast<Internal *>(m_root.get());
            assert(root != nullptr);
            assert(res == root->children[0].get());
            m_root = std::move(root->children[0]);
        }
    }
    else if (par->size < (Internal::capacity + 1) / 2) {
        if (!borrow(path)) {
            merge(path);
        }
    }

    return res;
}

template <class Key, class Value, std::size_t BlockSize, class Less>
auto BPTree<Key, Value, BlockSize, Less>::try_merge_parametric(Node * left, Node * right, Key delimiter)
        -> Node *
{
    Node * res = nullptr;

    Leaf * left_leaf = dynamic_cast<Leaf *>(left);
    Leaf * right_leaf = dynamic_cast<Leaf *>(right);
    if (left_leaf != nullptr && right_leaf != nullptr && right_leaf->size + left_leaf->size <= Leaf::capacity) {
        res = left_leaf;
        for (size_type i = left_leaf->size; i < left_leaf->size + right_leaf->size; ++i) {
            left_leaf->storage[i] = std::move(right_leaf->storage[i - left_leaf->size]);
        }
        left_leaf->size += right_leaf->size;
        right_leaf->size = 0;
        assert(left_leaf->size <= Internal::capacity);
        left_leaf->r_sibling = right_leaf->r_sibling;
    }
    else {
        Internal * right_internal = dynamic_cast<Internal *>(right);
        Internal * left_internal = dynamic_cast<Internal *>(left);
        if (left_internal != nullptr && right_internal != nullptr && right_internal->size + left_internal->size <= Internal::capacity) {
            res = left_internal;
            for (size_type i = left_internal->size + 1; i < left_internal->size + 1 + right_internal->size + 1; ++i) {
                left_internal->children[i] = std::move(right_internal->children[i - left_internal->size - 1]);
            }
            left_internal->keys[left_internal->size] = std::move(delimiter);
            left_internal->size++;
            for (size_type i = left_internal->size; i < left_internal->size + right_internal->size; ++i) {
                left_internal->keys[i] = std::move(right_internal->keys[i - left_internal->size]);
            }
            left_internal->size += right_internal->size;
            right_internal->size = 0;
            assert(left_internal->size <= Internal::capacity);
        }
    }

    return res;
}

template <class Key, class Value, std::size_t BlockSize, class Less>
auto BPTree<Key, Value, BlockSize, Less>::left(const std::unique_ptr<Node> & cur)
        -> Leaf *
{
    return left(cur.get());
}

template <class Key, class Value, std::size_t BlockSize, class Less>
auto BPTree<Key, Value, BlockSize, Less>::left(Node * cur)
        -> Leaf *
{
    Leaf * cur_leaf = dynamic_cast<Leaf *>(cur);
    if (cur_leaf != nullptr) {
        return cur_leaf;
    }
    Internal * cur_internal = dynamic_cast<Internal *>(cur);
    if (cur_internal == nullptr) {
        throw std::logic_error("Unknown node type");
    }

    return left(cur_internal->children[0].get());
}

template <class Key, class Value, std::size_t BlockSize, class Less>
template <bool is_const, class T>
auto BPTree<Key, Value, BlockSize, Less>::lower_bound(T * self, const key_type & key)
        -> Iterator<is_const>
{
    std::vector<std::pair<std::conditional_t<is_const, const Node, Node> *, size_type>> path;
    path.emplace_back(self->m_root.get(), 0);
    using leaf_type = std::conditional_t<is_const, const Leaf, Leaf>;
    using iterator_type = std::conditional_t<is_const, const std::pair<Key, Value> *, std::pair<Key, Value> *>;
    leaf_type * target = find_leaf<is_const>(path, key);
    iterator_type it = std::lower_bound(target->storage.begin(), target->storage.begin() + target->size, key, comp_pair);

    if (it == target->storage.begin() + target->size) {
        if (target->r_sibling != nullptr) {
            return {target->r_sibling->storage.begin(), target->r_sibling, *self};
        }
        return self->end();
    }
    return Iterator<is_const>(it, target, *self);
}

template <class Key, class Value, std::size_t BlockSize, class Less>
template <bool is_const>
auto BPTree<Key, Value, BlockSize, Less>::upper_bound(const key_type & key)
        -> Iterator<is_const>
{
    Iterator<is_const> lb = lower_bound(key);
    if (lb != end() && lb->first == key) {
        return ++lb;
    }
    return lb;
}
