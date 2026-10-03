# Phase 3 — Frame Processing Pipeline

- [x] 3.1 Build frame capture pipeline
- [x] 3.2 Implement frame sampling
- [x] 3.3 Resize frames for AI inference
- [x] 3.4 Add timestamp and frame ID
- [x] 3.5 Add camera ID to every frame
- [x] 3.6 Implement image-quality checks
- [x] 3.7 Detect frozen/black/very dark/blurred frames
- [x] 3.8 Add FPS and processing-latency logging
- [x] 3.9 Build common AI pipeline interface

## Phase 3 design

CameraManager -> FramePacket -> FramePipeline -> AIFrame -> AIProcessor

The default AIProcessor is a NoOpAIProcessor. Real person, face, and infrastructure
processors are intentionally added in later phases and receive the same AIFrame interface.
