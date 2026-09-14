-- ============================================================
-- RESTAURANT SALES ANALYSIS & INTELLIGENCE PLATFORM
-- DATABASE SCHEMA
-- Database: SQLite
-- ============================================================

PRAGMA foreign_keys = ON;

-- ============================================================
-- 1. AREAS
-- ============================================================

DROP TABLE IF EXISTS order_items;
DROP TABLE IF EXISTS orders;
DROP TABLE IF EXISTS inventory;
DROP TABLE IF EXISTS dishes;
DROP TABLE IF EXISTS raw_materials;
DROP TABLE IF EXISTS suppliers;
DROP TABLE IF EXISTS restaurants;
DROP TABLE IF EXISTS areas;

CREATE TABLE areas (
    area_id TEXT PRIMARY KEY,
    area_name TEXT NOT NULL,
    city TEXT NOT NULL,
    state TEXT NOT NULL,
    population INTEGER
);

-- ============================================================
-- 2. RESTAURANTS
-- ============================================================

CREATE TABLE restaurants (
    restaurant_id TEXT PRIMARY KEY,
    restaurant_name TEXT NOT NULL,
    area_id TEXT NOT NULL,
    restaurant_type TEXT,

    FOREIGN KEY (area_id)
        REFERENCES areas(area_id)
);

-- ============================================================
-- 3. DISHES
-- ============================================================

CREATE TABLE dishes (
    dish_id TEXT PRIMARY KEY,
    dish_name TEXT NOT NULL,
    category TEXT NOT NULL,
    selling_price REAL NOT NULL,
    ingredient_cost REAL NOT NULL,
    demand_weight REAL,

    CHECK (selling_price >= 0),
    CHECK (ingredient_cost >= 0)
);

-- ============================================================
-- 4. SUPPLIERS
-- ============================================================

CREATE TABLE suppliers (
    supplier_id TEXT PRIMARY KEY,
    supplier_name TEXT NOT NULL,
    supplier_city TEXT,
    rating REAL
);

-- ============================================================
-- 5. RAW MATERIALS
-- ============================================================

CREATE TABLE raw_materials (
    material_id TEXT PRIMARY KEY,
    material_name TEXT NOT NULL,
    category TEXT,
    shelf_life_days INTEGER,
    unit_cost REAL NOT NULL,
    reorder_level REAL NOT NULL,
    supplier_id TEXT,

    FOREIGN KEY (supplier_id)
        REFERENCES suppliers(supplier_id),

    CHECK (unit_cost >= 0),
    CHECK (reorder_level >= 0)
);

-- ============================================================
-- 6. ORDERS
-- ============================================================

CREATE TABLE orders (
    order_id TEXT PRIMARY KEY,
    order_datetime TEXT NOT NULL,
    order_date TEXT NOT NULL,
    customer_id TEXT,
    restaurant_id TEXT NOT NULL,
    area_id TEXT NOT NULL,
    order_value REAL NOT NULL,
    discount REAL DEFAULT 0,
    payment_method TEXT,
    order_status TEXT,
    promotion TEXT,

    FOREIGN KEY (restaurant_id)
        REFERENCES restaurants(restaurant_id),

    FOREIGN KEY (area_id)
        REFERENCES areas(area_id),

    CHECK (order_value >= 0),
    CHECK (discount >= 0)
);

-- ============================================================
-- 7. ORDER ITEMS
-- ============================================================

CREATE TABLE order_items (
    order_item_id TEXT PRIMARY KEY,
    order_id TEXT NOT NULL,
    dish_id TEXT NOT NULL,
    quantity INTEGER NOT NULL,
    selling_price REAL NOT NULL,
    discount REAL DEFAULT 0,
    revenue REAL NOT NULL,
    ingredient_cost REAL NOT NULL,
    profit REAL NOT NULL,

    FOREIGN KEY (order_id)
        REFERENCES orders(order_id),

    FOREIGN KEY (dish_id)
        REFERENCES dishes(dish_id),

    CHECK (quantity > 0),
    CHECK (selling_price >= 0),
    CHECK (discount >= 0),
    CHECK (revenue >= 0)
);

-- ============================================================
-- 8. INVENTORY
-- ============================================================

CREATE TABLE inventory (
    inventory_id TEXT PRIMARY KEY,
    material_id TEXT NOT NULL,
    area_id TEXT NOT NULL,
    purchase_date TEXT NOT NULL,
    quantity_purchased REAL NOT NULL,
    quantity_used REAL NOT NULL,
    quantity_wasted REAL NOT NULL,
    unit_cost REAL NOT NULL,
    expiry_date TEXT,

    FOREIGN KEY (material_id)
        REFERENCES raw_materials(material_id),

    FOREIGN KEY (area_id)
        REFERENCES areas(area_id),

    CHECK (quantity_purchased >= 0),
    CHECK (quantity_used >= 0),
    CHECK (quantity_wasted >= 0),
    CHECK (unit_cost >= 0)
);

-- ============================================================
-- INDEXES
-- ============================================================

CREATE INDEX idx_orders_date
ON orders(order_date);

CREATE INDEX idx_orders_restaurant
ON orders(restaurant_id);

CREATE INDEX idx_orders_area
ON orders(area_id);

CREATE INDEX idx_order_items_order
ON order_items(order_id);

CREATE INDEX idx_order_items_dish
ON order_items(dish_id);

CREATE INDEX idx_inventory_material
ON inventory(material_id);

CREATE INDEX idx_inventory_area
ON inventory(area_id);

CREATE INDEX idx_inventory_expiry
ON inventory(expiry_date);

-- ============================================================
-- SCHEMA COMPLETE
-- ============================================================