# Fall Detection & Caregiver Alert System

Real-time fall detection using phone accelerometer data, ML classification, and
live caregiver alerts. Built for elderly care settings (piloted at Manasa Old Age Home).

## Architecture

```
[Phone Browser]  --sensor data-->  [Flask ML Service]  --prediction-->  [Node Backend]
DeviceMotion API      (Python)      (scikit-learn)         (Express)         |
                                                                        Socket.io push
                                                                              |
                                                                              v
                                                                    [React Dashboard]
                                                                    (caregiver view)
```

**Why two backends, not one?** Python has the mature ML ecosystem (scikit-learn,
pandas, numpy) needed for signal processing and model training. Node handles
real-time bidirectional communication (Socket.io) and general API/auth logic
where its event-loop model shines. Splitting them is a deliberate systems
decision, not accidental complexity.

## Components

- `ml-service/` — Flask app. Trains and serves the fall classification model.
- `backend/` — Express + MongoDB. Auth, event logging, Socket.io alert broadcasting.
- `frontend/` — React caregiver dashboard + sensor-capture web page.
- `docs/` — dataset notes, model performance logs, pilot findings.

## Status
- [ ] Dataset downloaded & explored
- [ ] Feature extraction implemented
- [ ] Model trained & evaluated
- [ ] Flask /predict endpoint working
- [ ] Node backend + Socket.io
- [ ] React dashboard
- [ ] End-to-end integration
- [ ] Pilot test
