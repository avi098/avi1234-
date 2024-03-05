import React, { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import Validation from "./LoginValidation";
import axios from "axios";

function handleclick() {
  alert("Go to Farmer-Signup to create the account");
}

function hello() {
  alert("loged-in successfully");
}

export default function Login() {
  const [values, setValues] = useState({
    email: "",
    password: "",
  });
  const navigate = useNavigate();

  const handleInput = (event) => {
    setValues((prev) => ({
      ...prev,
      [event.target.name]: event.target.value,
    }));
  };

  const [errors, setErrors] = useState({});
  const [backendError, setBackendError] = useState([]);

  const handleSubmit = (event) => {
    event.preventDefault();
    const err = Validation(values);
    setErrors(err);
    if (err.email === "" && err.password === "") {
      axios
        .post("http://localhost:8081/login", values)
        .then((res) => {
          if (res.data.errors) {
            setBackendError(res.data.errors);
          } else {
            setBackendError([]);
            if (res.data === "Success") {
              navigate("/about");
            } else {
              alert("No record existed");
            }
          }
        })
        .catch((err) => console.log(err));
    }
  };

  return (
    <>
      <br />
      <br />
      <div className="d-flex justify-content-center align-items-center bg-dark vh-100">
        <div className="bg-white p-3 rounded w-30">
          <h4>Sign-In</h4>
          {backendError ? (
            backendError.map((e) => <p className="text-danger">{e.msg}</p>)
          ) : (
            <span></span>
          )}
          <form action="" onSubmit={handleSubmit}>
            <div className="mb-3">
              <label htmlFor="email">
                <strong>Email</strong>
              </label>
              <input
                type="email"
                placeholder="Enter Email"
                name="email"
                onChange={handleInput}
                className="form-control rounded-0"
              />
              {errors.email && (
                <span className="text-danger">{errors.email}</span>
              )}
            </div>
            <div className="mb-3">
              <label htmlFor="password">
                <strong>Password</strong>
              </label>
              <input
                type="password"
                placeholder="Enter Password"
                name="password"
                onChange={handleInput}
                className="form-control rounded-0"
              />
              {errors.password && (
                <span className="text-danger">{errors.password}</span>
              )}
            </div>
            <center>
              <button
                type="submit"
                onClick={hello}
                className="btn btn-success rounded-0"
              >
                Log in
              </button>
            </center>
            <p>You are agree to our terms and policies</p>
            <center>
              <Link
                to="/pricing"
                className="btn btn-default border bg-light rounded-0 text-decoration-none"
                onClick={handleclick}
                id="hello"
              >
                Create Account
              </Link>
            </center>
          </form>
        </div>
      </div>
    </>
  );
}
