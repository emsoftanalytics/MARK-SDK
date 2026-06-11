# License Boundary

## MIT: `mark`

The local Python package in this directory is MIT licensed.

It should include:

- public data types
- local memory storage
- local retrieval fallback
- local safety constraints
- skill and plugin interfaces
- local developer tutorials
- optional local adapters that do not contain cloud service logic

## Separately Licensed: MARK Cloud

Cloud service implementations live outside this package. The MIT package
calls them only through stable plugin hooks or cloud clients.

The rule is simple:

- The SDK may define the slot.
- The SDK may define the request and response shape.
- The SDK may provide a small local fallback.
- The SDK must not ship cloud implementation logic.

## Practical Placement

- `sdk/python/mark`: MIT local SDK.
- `sdk/python/mark-adapters`: MIT framework adapters.
- `sdk/python/mark-cloud`: cloud client package. Client-side request code may be
  open, but hosted service implementations stay in the cloud.
- Hosted services: cloud service code and data systems.
