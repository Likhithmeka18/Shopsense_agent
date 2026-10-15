# ShopSense Refund Policy

Effective 2026-01-01. Governs how refunds are calculated, approved, and issued once a return or refund request is accepted.

## RF-1 Refund Amount

### RF-1.1 Standard Refund

The customer receives the full item price paid, excluding the original shipping fee -- unless the return reason is RP-1.3 (defective or wrong item), in which case shipping is also refunded.

### RF-1.2 Partial Refunds

If an item is returned without original packaging or with visible wear not disclosed at the time of the return request, ShopSense may deduct up to 20% as a restocking adjustment, subject to the RF-3 approval tiers below.

## RF-2 Refund Timing

| Refund Method | Processing Time After Item Received |
|---|---|
| Original payment method (card) | 5-7 business days |
| UPI / wallet | 2-3 business days |
| Store credit | Instant |
| Cash on Delivery orders (bank transfer) | 7-10 business days |

Refund timing is measured from when the returned item passes warehouse quality check. Warehouse QC itself must complete within 48 hours of receipt -- internal SLA code **SLA-48**.

## RF-3 Approval Tiers

Refunds are approved automatically or escalated based on amount:

| Refund Amount (INR) | Approval Tier |
|---|---|
| Below 2,000 | Auto-approved, no manual review |
| 2,000 to 9,999 | Manager review required |
| 10,000 and above | Executive review required |

A customer account flagged for return abuse (3 or more high-value returns of INR 5,000+ within 90 days; definition in fraud_prevention.md FP-1) is routed to manual review regardless of amount -- see escalation code **ESC-ABUSE** in the Escalation SOP. Who may approve each tier, and AI-agent limits, are in refund_authorisation_matrix.md RA-1.

## RF-4 Refund Request Without a Return

### RF-4.1 Order Never Arrived

If tracking shows no movement for 10 days past the estimated delivery date, the customer may request a full refund without returning anything, since nothing was received. See also Shipping Policy section SP-4.1 (lost shipments, escalation code ESC-LOST).

### RF-4.2 Duplicate Charge

A duplicate charge for the same order is refunded in full within 3 business days once confirmed by finance, independent of the RF-3 approval tiers above.

## RF-5 Related Policies

See the Return Policy (document: return_policy.md) for return windows and eligibility, and the Shipping Policy (document: shipping_policy.md) for the delivery SLAs referenced in RF-4.1.
