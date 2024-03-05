import "./App.css";
import Navbar from "./components/Navbar";
import Home from "./components/Home";
import Customer from "./components/Customer";
import Farmer from "./components/Farmer";
import { Route, Routes } from "react-router-dom";
import Login from "./components/Login";

function App() {
  //App.js is a main file
  return (
    <>
      <Navbar />
      <u />
      <div className="container">
        <Routes>
          <Route path="/" element={<Home />} />
          <Route path="/pricing" element={<Farmer />} />
          <Route path="/login" element={<Login />} />
          <Route path="/about" element={<Customer />} />
        </Routes>
      </div>
    </>
  );
}

export default App;
