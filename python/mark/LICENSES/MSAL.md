# MARK Source Available License Boundary

This file documents the intended location of MARK's source-available boundary.
It is not applied to the `mark` Python SDK package.

The `mark` Python SDK is MIT licensed. It contains local developer runtime
interfaces, fallback implementations, safety schemas, plugin contracts, and
developer examples.

MARK Cloud implementations live in separate packages or services, such as:

- `mark-core`
- `mark-cloud`
- hosted MARK Cloud services
- private cloud plugins registered through `mark.plugins.PluginRegistry`

Those packages may use the MARK Source Available License or another proprietary
license chosen by the MARK maintainers. They must not be copied into this MIT
SDK package.

All hosted MARK Cloud service implementations — together with their
calibration, trained models, production data, and service logic — belong
outside the MIT SDK.

The MIT SDK may expose stable hook names and data contracts for cloud
services, but never their implementations.
