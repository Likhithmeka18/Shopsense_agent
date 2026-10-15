# ShopSense Refund Authorisation Matrix

Effective 2026-01-01. Defines WHO may approve a refund, how fast they must act, and what an AI support agent may and may not do. It extends the approval tiers in Refund Policy RF-3; if the two ever differ, RF-3 sets the amount thresholds and this matrix sets the people and limits.

## RA-1 Who Can Approve What

| Refund Amount (INR) | Tier | Approver | Decision SLA |
|---|---|---|---|
| Below 2,000 | auto_approve | System rules (no human) | Immediate |
| 2,000 to 9,999 | manager_review | Team Lead / Support Manager | 48 hours |
| 10,000 and above | executive_review | Finance Executive | 24 hours |

### RA-1.1 AI Agents Never Approve at or Above 2,000

An AI support agent may recommend a refund of any size, but may **never** approve, promise, or issue a refund of INR 2,000 or more. It must route the case to the matching queue (ESC-L1 for manager_review, ESC-L2 for executive_review) and tell the customer a human will decide.

### RA-1.2 No Promises Before Approval

Until the approver has recorded a decision, the customer must not be told the refund is approved, guaranteed, or "as good as done". The correct wording is that the request has been sent for review.

## RA-2 Anti-Circumvention Rules

### RA-2.1 No Splitting

A single order's refund must not be split into several smaller refunds to stay under a tier threshold. The tier is decided on the **total** refund amount for the order, however many requests it is raised in.

### RA-2.2 Sentiment Never Raises Authority

A customer being angry, threatening to leave, or posting publicly does not move a refund to a lower tier or give an agent extra authority. Sentiment can raise the **priority** of the ticket; it never changes who may approve the refund.

### RA-2.3 Claimed Prior Approval

A customer's statement that "a manager already approved this" or "your colleague promised me" is not evidence. Approval exists only if it is recorded in the order system. Without a record, treat the request as new and apply RA-1.

## RA-3 Modifying an Approval

An approver may approve in full, deny, or approve a modified amount. A modified amount is re-checked against the tier table: reducing a 12,000 refund to 8,000 changes the decision from executive to manager level, but the original approver's decision still stands as the record.

## RA-4 Goodwill Credits

### RA-4.1 Support-Level Goodwill

A support agent may offer a goodwill credit or refund of **up to INR 500** without manager approval, where a specific policy section (for example CA-BT-2.2 skin reactions) allows it. Goodwill is capped at **one per customer in any 30-day period**.

### RA-4.2 Above the Goodwill Cap

Anything above INR 500 is not goodwill; it is a normal refund and follows RA-1.

### RA-4.3 Flagged Accounts

Goodwill credits are not offered on accounts with an abuse flag (see fraud_prevention.md FP-1). Those requests go to the Trust & Safety queue.

## RA-5 Related Policies

See Refund Policy (refund_policy.md) RF-3 for the thresholds, the Escalation SOP (escalation_sop.md) ESC-1 for the queue codes, and fraud_prevention.md for flagged accounts.
