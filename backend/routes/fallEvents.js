const express = require("express");
const router = express.Router();
const FallEvent = require("../models/FallEvent");

// POST /api/events -- log a new fall prediction and push it live to dashboards
router.post("/", async (req, res) => {
  try {
    const { patientId, fallDetected, confidence } = req.body;

    const event = new FallEvent({ patientId, fallDetected, confidence });
    await event.save();

    // Only push a real-time alert if a fall was actually detected --
    // no need to spam the dashboard for routine "no fall" checks.
    if (fallDetected) {
      const io = req.app.get("io");
      io.emit("fallAlert", event);
    }

    res.status(201).json(event);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// GET /api/events -- fetch recent events (for the dashboard's history view)
router.get("/", async (req, res) => {
  try {
    const events = await FallEvent.find().sort({ timestamp: -1 }).limit(50);
    res.json(events);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

// PATCH /api/events/:id/acknowledge -- caregiver marks an alert as handled
router.patch("/:id/acknowledge", async (req, res) => {
  try {
    const event = await FallEvent.findByIdAndUpdate(
      req.params.id,
      { acknowledged: true, acknowledgedAt: new Date() },
      { new: true }
    );
    res.json(event);
  } catch (err) {
    res.status(500).json({ error: err.message });
  }
});

module.exports = router;
