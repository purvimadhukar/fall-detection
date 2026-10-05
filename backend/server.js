const express = require("express");
const mongoose = require("mongoose");
const http = require("http");
const { Server } = require("socket.io");
const cors = require("cors");
require("dotenv").config();

const app = express();
app.use(cors());
app.use(express.json()); // lets us read JSON request bodies

// Create an HTTP server manually (instead of app.listen directly) because
// Socket.io needs to attach to the raw HTTP server, not just Express.
const server = http.createServer(app);
const io = new Server(server, {
  cors: { origin: "*" }, // fine for development; tighten this before real deployment
});

// Connect to MongoDB
mongoose.connect(process.env.MONGO_URI)
  .then(() => console.log("MongoDB connected"))
  .catch((err) => console.error("MongoDB connection error:", err));

// Make `io` accessible inside route files via req.app.get("io")
app.set("io", io);

// Routes
const fallEventRoutes = require("./routes/fallEvents");
app.use("/api/events", fallEventRoutes);

// Socket.io: log when a dashboard connects/disconnects
io.on("connection", (socket) => {
  console.log("Dashboard connected:", socket.id);
  socket.on("disconnect", () => {
    console.log("Dashboard disconnected:", socket.id);
  });
});

app.get("/health", (req, res) => res.json({ status: "ok" }));

const PORT = process.env.PORT || 5000;
server.listen(PORT, () => console.log(`Server running on port ${PORT}`));
