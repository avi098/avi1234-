import React from "react";
import "./Home.css";
import { Link, useMatch, useResolvedPath } from "react-router-dom";

export default function Navbar() {
  return (
    <>
      <nav className="navbar navbar-expand-lg bg-body-tertiary fixed-top">
        <div className="container">
          <Link to="/" className="navbar-brand">
            <img
              src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSX2wtLknZl98RnMkd5X6wHBAg92nj4So7lqoSw1Hh0FQ&s"
              alt="Bootstrap"
              width="30"
              height="24"
            />
            <span className="text-warning">GO</span>FRESH
          </Link>
          <button
            className="navbar-toggler"
            type="button"
            data-bs-toggle="collapse"
            data-bs-target="#navbarSupportedContent"
            aria-controls="navbarSupportedContent"
            aria-expanded="false"
            aria-label="Toggle navigation"
          >
            <span className="navbar-toggler-icon"></span>
          </button>
          <div className="collapse navbar-collapse" id="navbarSupportedContent">
            <ul className="navbar-nav ms-auto mb-2 mb-lg-0">
              <li className="nav-item">
                <CustomLink
                  to="/pricing"
                  className="nav-link active"
                  aria-current="page"
                >
                  <button type="button" className="btn btn-outline-primary">
                    Farmer-Signup
                  </button>
                </CustomLink>
              </li>
              <li className="nav-item">
                <CustomLink
                  to="/login"
                  className="nav-link active"
                  aria-current="page"
                >
                  <button type="button" className="btn btn-outline-primary">
                    Farmer-Login
                  </button>
                </CustomLink>
              </li>
              <li className="nav-item">
                <CustomLink
                  to="/about"
                  className="nav-link"
                  aria-current="page"
                >
                  <button type="button" className="btn btn-outline-primary">
                    Products
                  </button>
                </CustomLink>
              </li>
            </ul>
          </div>
        </div>
      </nav>
    </>
  );
}

function CustomLink({ to, children, ...props }) {
  const resolvedPath = useResolvedPath(to);
  const isActive = useMatch({ path: resolvedPath.pathname, end: true });
  return (
    <li className={isActive ? "active" : ""}>
      <Link to={to} {...props}>
        {children}
      </Link>
    </li>
  );
}
