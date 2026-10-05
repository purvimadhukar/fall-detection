const mongoose = require("mongoose");

const fallEventSchema = new mongoose.Schema({
  patientId: { type: mongoose.Schema.Types.ObjectId, ref: "User", required: true },
  fallDetected: { type: Boolean, required: true },
  confidence: { type: Number, required: true },
  timestamp: { type: Date, default: Date.now },
  acknowledged: { type: Boolean, default: false },
  acknowledgedAt: { type: Date },
});

module.exports = mongoose.model("FallEvent", fallEventSchema);
