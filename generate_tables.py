"""
ShopSense - Asset C generator: mock order/customer/product/shipment/refund tables.
Deterministic (seeded) so tickets generated separately can reference these same IDs.

Two layers, deliberately kept separate:
  * BASE (60 customers / 120 orders / 15 refunds): produced by the global RNG exactly as in
    the first build. Tests, demos and docs reference specific base IDs (ORD-4099, CUST-0050 ...),
    so this layer must stay byte-identical -- never reorder the calls that consume `random`.
  * EXTENSION (extend_tables): +90 customers / +480 orders with its OWN RNG (SEED + 1), APPEND-ONLY.
    Extension orders belong only to extension customers (CUST-0061+), so no base row is touched.
    It adds what the Data Sourcing Guidelines ask of Asset C but the base sample is too small to
    contain: customers who really meet the return-abuse definition (3+ returns of INR 5,000+
    inside 90 days, see policies/fraud_prevention.md FP-1), near-miss customers (2 such returns,
    NOT flagged), many pending manager/executive refunds, several >= INR 10,000, and shipments
    stuck long enough to count as lost (SP-4.1).
"""
import csv
import json
import random
from datetime import datetime, timedelta

SEED = 42
random.seed(SEED)

OUT_DIR = "data/shopsense/mock_api"
N_CUSTOMERS = 60
N_PRODUCTS = 40
N_ORDERS = 120

# Extension layer (append-only, own RNG so the base layer above never changes)
N_EXT_CUSTOMERS = 90
N_EXT_ORDERS = 480
N_EXT_ABUSERS = 9        # meet the FP-1 definition and carry abuse_flag=True
N_EXT_NEAR_MISS = 4      # exactly 2 high-value returns in 90 days: NOT flagged
N_EXT_BIG_PENDING = 8    # delivered orders >= INR 10,000 with a pending refund request
N_EXT_LOST = 6           # shipped, no movement for 10+ days
HIGH_VALUE_INR = 5000

FIRST_NAMES = ["Aarav", "Priya", "Rohan", "Ananya", "Vikram", "Sneha", "Arjun", "Divya",
               "Kiran", "Meera", "Rahul", "Pooja", "Sanjay", "Neha", "Amit", "Ritu",
               "Karan", "Isha", "Varun", "Shreya"]
LAST_NAMES = ["Sharma", "Verma", "Reddy", "Nair", "Iyer", "Gupta", "Mehta", "Rao",
              "Joshi", "Kapoor", "Menon", "Pillai", "Chawla", "Desai", "Bhatt"]

CATEGORIES = {
    "electronics": ["Wireless Earbuds", "Bluetooth Speaker", "Power Bank 20000mAh",
                    "USB-C Charger", "Smartwatch", "Laptop Sleeve", "Mechanical Keyboard"],
    "apparel": ["Cotton T-Shirt", "Denim Jacket", "Running Shoes", "Wool Sweater",
                "Formal Shirt", "Yoga Pants", "Rain Jacket"],
    "groceries": ["Basmati Rice 5kg", "Organic Honey 500g", "Green Tea Pack",
                  "Almonds 1kg", "Cold Pressed Oil 1L"],
    "home": ["Non-stick Pan Set", "LED Desk Lamp", "Memory Foam Pillow",
             "Ceramic Mug Set", "Storage Organizer Box"],
    "beauty": ["Vitamin C Serum", "Sunscreen SPF50", "Hair Dryer", "Lip Balm Set"],
}

CARRIERS = ["BlueDart", "Delhivery", "DTDC", "Ecom Express", "India Post"]
SCAN_STATES = ["label_created", "picked_up", "in_transit", "out_for_delivery",
               "delivered", "delivery_exception", "returned_to_sender"]

REFERENCE_CLOCK = datetime(2026, 10, 1, 9, 0, 0)  # frozen "now" for reproducibility


def rand_date_within(days_back_min, days_back_max):
    d = random.randint(days_back_min, days_back_max)
    return REFERENCE_CLOCK - timedelta(days=d)


def gen_customers():
    rows = []
    for i in range(1, N_CUSTOMERS + 1):
        fn, ln = random.choice(FIRST_NAMES), random.choice(LAST_NAMES)
        cid = f"CUST-{i:04d}"
        signup = rand_date_within(30, 900)
        # Most customers are fine; a minority have return-abuse flags for guardrail testing
        return_rate = round(random.betavariate(2, 10), 3)  # skewed low, long tail
        abuse_flag = return_rate > 0.35
        rows.append({
            "customer_id": cid,
            "name": f"{fn} {ln}",
            "email": f"{fn.lower()}.{ln.lower()}{i}@example.com",
            "city": random.choice(["Hyderabad", "Bengaluru", "Mumbai", "Pune", "Chennai",
                                    "Delhi", "Kolkata", "Ahmedabad", "Jaipur", "Kochi"]),
            "signup_date": signup.strftime("%Y-%m-%d"),
            "total_orders": 0,  # filled in after orders generated
            "return_rate": return_rate,
            "abuse_flag": abuse_flag,
        })
    return rows


def gen_products():
    rows = []
    pid = 1
    for category, items in CATEGORIES.items():
        for name in items:
            base_price = {
                "electronics": random.randint(800, 15000),
                "apparel": random.randint(400, 3500),
                "groceries": random.randint(150, 1200),
                "home": random.randint(300, 4000),
                "beauty": random.randint(200, 2000),
            }[category]
            rows.append({
                "product_id": f"PROD-{pid:04d}",
                "name": name,
                "category": category,
                "price_inr": base_price,
                "return_window_days": {
                    "electronics": 15, "apparel": 45, "groceries": 0,
                    "home": 30, "beauty": 10,
                }[category],
            })
            pid += 1
            if pid > N_PRODUCTS:
                return rows
    return rows


def gen_orders(customers, products):
    rows = []
    order_items_rows = []
    for i in range(1, N_ORDERS + 1):
        oid = f"ORD-{4000 + i}"
        cust = random.choice(customers)
        order_date = rand_date_within(1, 120)
        n_items = random.randint(1, 3)
        chosen = random.sample(products, n_items)
        total = sum(p["price_inr"] for p in chosen)

        # Status distribution: mostly delivered, some in progress, few cancelled/returned
        status = random.choices(
            ["delivered", "shipped", "processing", "cancelled", "returned"],
            weights=[60, 15, 10, 8, 7],
        )[0]

        rows.append({
            "order_id": oid,
            "customer_id": cust["customer_id"],
            "order_date": order_date.strftime("%Y-%m-%d"),
            "status": status,
            "total_amount_inr": total,
            "payment_method": random.choice(["UPI", "Credit Card", "Debit Card", "Cash on Delivery", "Net Banking"]),
        })

        for p in chosen:
            order_items_rows.append({
                "order_id": oid,
                "product_id": p["product_id"],
                "product_name": p["name"],
                "category": p["category"],
                "unit_price_inr": p["price_inr"],
                "quantity": 1,
            })

    # backfill customer total_orders
    counts = {}
    for r in rows:
        counts[r["customer_id"]] = counts.get(r["customer_id"], 0) + 1
    for c in customers:
        c["total_orders"] = counts.get(c["customer_id"], 0)

    return rows, order_items_rows


def gen_shipments(orders):
    rows = []
    for o in orders:
        if o["status"] in ("processing",):
            continue  # not shipped yet
        sid = f"SHIP-{o['order_id'].split('-')[1]}"
        if o["status"] == "cancelled":
            scan_state = "cancelled_pre_ship"
        elif o["status"] == "returned":
            scan_state = "returned_to_sender"
        elif o["status"] == "delivered":
            scan_state = "delivered"
        else:
            scan_state = random.choice(["picked_up", "in_transit", "out_for_delivery", "delivery_exception"])

        ship_date = datetime.strptime(o["order_date"], "%Y-%m-%d") + timedelta(days=random.randint(1, 3))
        rows.append({
            "shipment_id": sid,
            "order_id": o["order_id"],
            "carrier": random.choice(CARRIERS),
            "ship_date": ship_date.strftime("%Y-%m-%d"),
            "scan_state": scan_state,
            "tracking_number": f"TRK{random.randint(10**9, 10**10 - 1)}",
        })
    return rows


def gen_refunds(orders, order_items, customers):
    cust_by_id = {c["customer_id"]: c for c in customers}
    items_by_order = {}
    for it in order_items:
        items_by_order.setdefault(it["order_id"], []).append(it)

    rows = []
    rid = 1
    for o in orders:
        if o["status"] != "returned" and random.random() > 0.12:
            continue  # only some non-returned orders also have a refund request
        if o["status"] != "returned" and o["status"] not in ("delivered",):
            continue

        items = items_by_order.get(o["order_id"], [])
        if not items:
            continue
        refund_amount = sum(it["unit_price_inr"] for it in items)

        # Approval tiers: auto-approve under 2000, manager review 2000-10000, exec above 10000
        if refund_amount < 2000:
            tier = "auto"
            approved = True
        elif refund_amount < 10000:
            tier = "manager_review"
            approved = random.random() > 0.2
        else:
            tier = "executive_review"
            approved = random.random() > 0.4

        req_date = datetime.strptime(o["order_date"], "%Y-%m-%d") + timedelta(days=random.randint(2, 20))
        rows.append({
            "refund_id": f"REF-{rid:04d}",
            "order_id": o["order_id"],
            "customer_id": o["customer_id"],
            "amount_inr": refund_amount,
            "approval_tier": tier,
            "approved": approved,
            "requested_date": req_date.strftime("%Y-%m-%d"),
            "reason": random.choice([
                "item_defective", "wrong_item_received", "changed_mind",
                "late_delivery", "item_not_as_described", "duplicate_order",
            ]),
        })
        rid += 1
    return rows


# --------------------------------------------------------------------------
# EXTENSION LAYER (append-only; own RNG; see module docstring)
# --------------------------------------------------------------------------
def _refund_tier(amount):
    """Same thresholds as shopsense.tools.order_tools.refund_tier_for_amount."""
    if amount < 2000:
        return "auto"
    if amount < 10000:
        return "manager_review"
    return "executive_review"


def extend_tables(customers, products, orders, order_items, shipments, refunds):
    """Append the extension layer to the six base lists (in place) and return
    {'customers': [...], 'orders': [...]} describing ONLY the new rows, with
    the extra context (categories, refund, flag) that the ticket generator
    needs to write realistic tickets against them."""
    rng = random.Random(SEED + 1)
    prod_by_id = {p["product_id"]: p for p in products}
    expensive = [p for p in products if p["price_inr"] >= 4000]
    all_products = list(products)

    def days_ago(n):
        return REFERENCE_CLOCK - timedelta(days=n)

    # ---- customers -------------------------------------------------------
    first_new = len(customers) + 1
    ext_customers = []
    for i in range(first_new, first_new + N_EXT_CUSTOMERS):
        fn, ln = rng.choice(FIRST_NAMES), rng.choice(LAST_NAMES)
        ext_customers.append({
            "customer_id": f"CUST-{i:04d}",
            "name": f"{fn} {ln}",
            "email": f"{fn.lower()}.{ln.lower()}{i}@example.com",
            "city": rng.choice(["Hyderabad", "Bengaluru", "Mumbai", "Pune", "Chennai",
                                "Delhi", "Kolkata", "Ahmedabad", "Jaipur", "Kochi"]),
            "signup_date": days_ago(rng.randint(30, 900)).strftime("%Y-%m-%d"),
            "total_orders": 0,
            "return_rate": 0.0,
            "abuse_flag": False,
        })
    shuffled = ext_customers[:]
    rng.shuffle(shuffled)
    abusers = shuffled[:N_EXT_ABUSERS]
    near_miss = shuffled[N_EXT_ABUSERS:N_EXT_ABUSERS + N_EXT_NEAR_MISS]
    for c in abusers:
        c["abuse_flag"] = True

    # ---- order descriptors (what to create, before ids are assigned) ----
    descriptors = []   # dicts: customer, items, status, age_days, refund ("pending"|"decided"|None), kind

    def hv_items(min_total):
        while True:
            chosen = rng.sample(expensive, rng.randint(1, 2))
            if sum(p["price_inr"] for p in chosen) >= min_total:
                return chosen

    for c in abusers:                      # 4 returned high-value orders inside 90 days, latest one still pending
        ages = sorted(rng.sample(range(8, 86), 4))
        for k, age in enumerate(ages):
            descriptors.append({"customer": c, "items": hv_items(HIGH_VALUE_INR), "status": "returned",
                                "age": age, "refund": "pending" if k == 0 else "decided", "kind": "abuser_return"})
    for c in near_miss:                    # only 2 -> below the 3+ threshold, so NOT flagged
        for age in rng.sample(range(10, 80), 2):
            descriptors.append({"customer": c, "items": hv_items(HIGH_VALUE_INR), "status": "returned",
                                "age": age, "refund": "decided", "kind": "near_miss_return"})
    plain = [c for c in ext_customers if c not in abusers and c not in near_miss]
    for c in rng.sample(plain, N_EXT_BIG_PENDING):   # big-ticket delivered orders awaiting an approver
        descriptors.append({"customer": c, "items": hv_items(10000), "status": "delivered",
                            "age": rng.randint(4, 25), "refund": "pending", "kind": "big_pending"})
    for c in rng.sample(plain, N_EXT_LOST):          # shipped, then silence: SP-4.1 lost shipment
        descriptors.append({"customer": c, "items": rng.sample(all_products, rng.randint(1, 2)), "status": "shipped",
                            "age": rng.randint(14, 24), "refund": None, "kind": "lost_shipment", "stuck": True})

    while len(descriptors) < N_EXT_ORDERS:
        c = rng.choice(plain + abusers[:3] + near_miss[:2])
        status = rng.choices(["delivered", "shipped", "processing", "cancelled", "returned"],
                             weights=[62, 14, 9, 7, 3])[0]
        age = {"delivered": rng.randint(5, 120), "shipped": rng.randint(2, 12), "processing": rng.randint(0, 3),
               "cancelled": rng.randint(1, 60), "returned": rng.randint(12, 100)}[status]
        refund = None
        if status == "returned":
            refund = "pending" if rng.random() < 0.35 else "decided"
        elif status == "delivered" and rng.random() < 0.09:
            refund = "pending" if rng.random() < 0.40 else "decided"
        descriptors.append({"customer": c, "items": rng.sample(all_products, rng.randint(1, 3)), "status": status,
                            "age": age, "refund": refund, "kind": "regular"})
    rng.shuffle(descriptors)

    # ---- materialise ------------------------------------------------------
    next_order = 4000 + len(orders) + 1
    next_refund = len(refunds) + 1
    ext_orders = []
    new_orders, new_items, new_ships, new_refunds = [], [], [], []
    for d in descriptors:
        oid = f"ORD-{next_order}"
        next_order += 1
        order_date = days_ago(d["age"])
        total = sum(p["price_inr"] for p in d["items"])
        order = {
            "order_id": oid, "customer_id": d["customer"]["customer_id"],
            "order_date": order_date.strftime("%Y-%m-%d"), "status": d["status"], "total_amount_inr": total,
            "payment_method": rng.choice(["UPI", "Credit Card", "Debit Card", "Cash on Delivery", "Net Banking"]),
        }
        new_orders.append(order)
        for p in d["items"]:
            new_items.append({"order_id": oid, "product_id": p["product_id"], "product_name": p["name"],
                              "category": p["category"], "unit_price_inr": p["price_inr"], "quantity": 1})

        if d["status"] != "processing":
            if d["status"] == "cancelled":
                scan = "cancelled_pre_ship"
            elif d["status"] == "returned":
                scan = "returned_to_sender"
            elif d["status"] == "delivered":
                scan = "delivered"
            elif d.get("stuck"):
                scan = rng.choice(["in_transit", "delivery_exception"])
            else:
                scan = rng.choice(["picked_up", "in_transit", "out_for_delivery", "delivery_exception"])
            ship_date = order_date + timedelta(days=rng.randint(1, 2))
            new_ships.append({
                "shipment_id": f"SHIP-{oid.split('-')[1]}", "order_id": oid, "carrier": rng.choice(CARRIERS),
                "ship_date": ship_date.strftime("%Y-%m-%d"), "scan_state": scan,
                "tracking_number": f"TRK{rng.randint(10**9, 10**10 - 1)}",
            })

        refund_row = None
        if d["refund"]:
            tier = _refund_tier(total)
            approved = True if (tier == "auto" and d["refund"] == "decided") else (d["refund"] == "decided" and rng.random() > 0.15)
            if tier == "auto" and d["refund"] == "pending":
                approved = False
            req = order_date + timedelta(days=rng.randint(2, 12))
            req = min(req, REFERENCE_CLOCK - timedelta(days=0))
            refund_row = {
                "refund_id": f"REF-{next_refund:04d}", "order_id": oid, "customer_id": d["customer"]["customer_id"],
                "amount_inr": total, "approval_tier": tier, "approved": approved,
                "requested_date": req.strftime("%Y-%m-%d"),
                "reason": rng.choice(["item_defective", "wrong_item_received", "changed_mind",
                                      "late_delivery", "item_not_as_described", "duplicate_order"]),
            }
            next_refund += 1
            new_refunds.append(refund_row)
        ext_orders.append({**order, "categories": sorted({p["category"] for p in d["items"]}),
                           "products": [p["name"] for p in d["items"]],
                           "kind": d["kind"], "refund": refund_row})

    # ---- backfill per-customer stats from the NEW orders -------------------
    by_cust = {}
    for o in new_orders:
        by_cust.setdefault(o["customer_id"], []).append(o)
    for c in ext_customers:
        mine = by_cust.get(c["customer_id"], [])
        c["total_orders"] = len(mine)
        c["return_rate"] = round(sum(1 for o in mine if o["status"] == "returned") / len(mine), 3) if mine else 0.0

    customers.extend(ext_customers)
    orders.extend(new_orders)
    order_items.extend(new_items)
    shipments.extend(new_ships)
    refunds.extend(new_refunds)
    return {
        "customers": [{"customer_id": c["customer_id"], "name": c["name"], "abuse_flag": c["abuse_flag"],
                       "near_miss": c in near_miss} for c in ext_customers],
        "orders": ext_orders,
    }


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    customers = gen_customers()
    products = gen_products()
    orders, order_items = gen_orders(customers, products)
    shipments = gen_shipments(orders)
    refunds = gen_refunds(orders, order_items, customers)

    base_order_details = [dict(o) for o in orders]            # frozen BEFORE the extension (ticket generator's base pool)
    base_customer_refs = [{"customer_id": c["customer_id"], "name": c["name"]} for c in customers]
    ext = extend_tables(customers, products, orders, order_items, shipments, refunds)

    write_csv(f"{OUT_DIR}/customers.csv", customers)
    write_csv(f"{OUT_DIR}/products.csv", products)
    write_csv(f"{OUT_DIR}/orders.csv", orders)
    write_csv(f"{OUT_DIR}/order_items.csv", order_items)
    write_csv(f"{OUT_DIR}/shipments.csv", shipments)
    write_csv(f"{OUT_DIR}/refunds.csv", refunds)

    # Keep order IDs available for the ticket generator to reference
    # `order_ids` / `customers` / `orders_detail` keep describing the BASE layer only (the base
    # tickets are drawn from them and must not change); the extension has its own keys.
    with open(f"{OUT_DIR}/_order_id_pool.json", "w") as f:
        json.dump({
            "order_ids": [o["order_id"] for o in base_order_details],
            "customers": base_customer_refs,
            "orders_detail": base_order_details,
            "ext_customers": ext["customers"],
            "ext_orders": ext["orders"],
        }, f, indent=2)

    print(f"customers: {len(customers)}")
    print(f"products: {len(products)}")
    print(f"orders: {len(orders)}")
    print(f"order_items: {len(order_items)}")
    print(f"shipments: {len(shipments)}")
    print(f"refunds: {len(refunds)}")


if __name__ == "__main__":
    main()
