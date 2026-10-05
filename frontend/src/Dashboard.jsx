import { useState, useEffect } from "react";
import { io } from "socket.io-client";
import axios from "axios";

const BACKEND_URL = "http://localhost:5050";

function Dashboard() {
  const [events, setEvents] = useState([]);
  const [connected, setConnected] = useState(false);

  // Runs once when the component first loads
  useEffect(() => {
    // Step 1: fetch existing event history
    axios.get(`${BACKEND_URL}/api/events`)
      .then((res) => setEvents(res.data))
      .catch((err) => console.error("Failed to load events:", err));

    // Step 2: open a live connection for real-time alerts
    const socket = io(BACKEND_URL);

    socket.on("connect", () => {
      console.log("Connected to server, socket id:", socket.id);
      setConnected(true);
    });

    socket.on("disconnect", () => setConnected(false));

    // This is the key part -- listens for the exact event name
    // the backend emits: io.emit("fallAlert", event)
    socket.on("fallAlert", (newEvent) => {
      console.log("Real-time fall alert received:", newEvent);
      setEvents((prevEvents) => [newEvent, ...prevEvents]);
    });

    // Cleanup: close the socket when the component unmounts
    return () => socket.disconnect();
  }, []); // empty array = run once on mount, not on every re-render

  const handleAcknowledge = async (eventId) => {
    try {
      const res = await axios.patch(`${BACKEND_URL}/api/events/${eventId}/acknowledge`);
      setEvents((prevEvents) =>
        prevEvents.map((e) => (e._id === eventId ? res.data : e))
      );
    } catch (err) {
      console.error("Failed to acknowledge:", err);
    }
  };

  return (
    <div style={{ fontFamily: "sans-serif", padding: "2rem", maxWidth: "700px", margin: "0 auto" }}>
      <h1>Fall Detection Dashboard</h1>
      <p>
        Status:{" "}
        <span style={{ color: connected ? "green" : "red", fontWeight: "bold" }}>
          {connected ? "Connected (live)" : "Disconnected"}
        </span>
      </p>

      <h2>Events</h2>
      {events.length === 0 && <p>No events yet.</p>}

      {events.map((event) => (
        <div
          key={event._id}
          style={{
            border: "1px solid #ccc",
            borderRadius: "8px",
            padding: "1rem",
            marginBottom: "1rem",
            backgroundColor: event.fallDetected ? "#ffe5e5" : "#f5f5f5",
          }}
        >
          <p>
            <strong>{event.fallDetected ? "FALL DETECTED" : "No fall"}</strong>
            {" "}(confidence: {(event.confidence * 100).toFixed(0)}%)
          </p>
          <p style={{ fontSize: "0.85rem", color: "#555" }}>
            {new Date(event.timestamp).toLocaleString()}
          </p>
          {event.fallDetected && !event.acknowledged && (
            <button onClick={() => handleAcknowledge(event._id)}>
              Acknowledge
            </button>
          )}
          {event.acknowledged && <p style={{ color: "green" }}>✓ Acknowledged</p>}
        </div>
      ))}
    </div>
  );
}

export default Dashboard;
