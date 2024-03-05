import React from "react";
import "./Home.js";
import { Link, useNavigate } from "react-router-dom";

export default function Customer() {
  const navigate = useNavigate();
  function handleclick() {
    window.alert("First log-in process should be done to buy products");
    navigate("/login");
  }

  const PlantProductCard = ({ title, description, price, imageUrl }) => (
    <div className="col-md-4 mb-4">
      <div className="card h-100">
        <img
          src={imageUrl}
          alt={`${title} product`}
          className="card-img-top"
          style={{ height: "200px", objectFit: "cover" }}
        />
        <div className="card-body">
          <h5 className="card-title">{title}</h5>
          <p className="card-text">{description}</p>
        </div>
        <div className="card-footer">
          <p className="card-text">
            <strong>Price:</strong> ${price}
          </p>
          <button
            type="submit"
            className="btn btn-primary"
            onClick={handleclick}
          >
            Buy Now
          </button>
          <br />
          <br />
          <Link
            to="/login"
            className="btn btn-default border bg-light rounded-0 text-decoration-none"
            onClick={handleclick}
            id="hello"
          >
            Go to Log-in
          </Link>
        </div>
      </div>
    </div>
  );

  const plantProducts = [
    {
      title: "Tomato",
      description: "Fresh tomato",
      price: 19.99,
      imageUrl:
        "https://www.foodrepublic.com/img/gallery/13-things-you-didnt-know-about-tomatoes/intro-1684521109.jpg",
    },
    {
      title: "Beetroot",
      description: "Fresh beetroot",
      price: 24.99,
      imageUrl:
        "https://www.healthifyme.com/blog/wp-content/uploads/2022/10/shutterstock_721955095-1.jpg",
    },
    {
      title: "Beans",
      description: "Fresh potato",
      price: 29.99,
      imageUrl:
        "https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSALkk72XXEV3KpZA_ga1T8TtA2tut9j1FctvpykZr6nw&s",
    },
  ];

  return (
    <div className="container mt-5">
      <div className="row">
        {plantProducts.map((product, index) => (
          <PlantProductCard key={index} {...product} />
        ))}
      </div>
    </div>
  );
}
