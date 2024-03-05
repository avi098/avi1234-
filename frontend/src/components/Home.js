import React from "react";
import "./Home.css";
export default function Home() {
  return (
    <>
      <div id="carouselExampleCaptions" className="carousel slide">
        <div className="carousel-indicators">
          <button
            type="button"
            data-bs-target="#carouselExampleCaptions"
            data-bs-slide-to="0"
            className="active"
            aria-current="true"
            aria-label="Slide 1"
          ></button>
          <button
            type="button"
            data-bs-target="#carouselExampleCaptions"
            data-bs-slide-to="1"
            aria-label="Slide 2"
          ></button>
          <button
            type="button"
            data-bs-target="#carouselExampleCaptions"
            data-bs-slide-to="2"
            aria-label="Slide 3"
          ></button>
        </div>
        <div className="carousel-inner">
          <div className="carousel-item active">
            <img
              src="https://img.freepik.com/free-photo/HEALTHy-vegetables-wooden-table_1150-38014.jpg?size=626&ext=jpg&ga=GA1.2.1744893221.1659592805&semt=ais"
              className="d-block w-100"
              alt="..."
            />
            <div className="carousel-caption">
              <h5>
                <p>
                  <span className="text-warning">HEALTH</span>IFY - Fruits And
                  Vegetables
                </p>
              </h5>
              <p>
                <span className="text-warning">HEALTH</span>IFY has IFY fruits
                and vegetables which have been harvested from the farmers.
                Enhanced website with trustworthy products based on plant based
                cultivated by the farmers.
              </p>
              <p>
                <a href="/" className="btn btn-warning mt-3">
                  Learn More
                </a>
              </p>
            </div>
          </div>
          <div className="carousel-item">
            <img
              src="https://img.freepik.com/free-vector/delivery-service-with-masks-concept_23-2148498421.jpg?size=626&ext=jpg&ga=GA1.1.1744893221.1659592805&semt=ais"
              className="d-block w-100"
              alt="..."
            />
            <div className="carousel-caption">
              <h5>
                <p>
                  <span className="text-warning">HEALTH</span>IFY - FRUITS AND
                  VEGETABLES
                </p>
              </h5>
              <p>
                <span className="text-warning">HEALTH</span>IFY has IFY fruits
                and vegetables which have been harvested from the farmers.
                Fast-delivery and packaged mode of fruits and vegetables is
                highly enhanced in our website.
              </p>
              <p>
                <a href="/" className="btn btn-warning mt-3">
                  Learn More
                </a>
              </p>
            </div>
          </div>
          <div className="carousel-item">
            <img
              src="https://img.freepik.com/free-photo/assortment-vegetables-green-herbs-market-vegetables-basket_2829-14020.jpg?size=626&ext=jpg&ga=GA1.1.1744893221.1659592805&semt=ais"
              className="d-block w-100"
              alt="..."
            />
            <div className="carousel-caption">
              <h5>
                <p>
                  <span className="text-warning">HEALTH</span>IFY - FRUITS AND
                  VEGETABLES
                </p>
              </h5>
              <p>
                <span className="text-warning">HEALTH</span>IFY has IFY fruits
                and vegetables which have been harvested from the farmers.
                Fruits and vegetables are harvested from farmers and highly
                trust in HEALTH based content.
              </p>
              <p>
                <a href="/" className="btn btn-warning mt-3">
                  Learn More
                </a>
              </p>
            </div>
          </div>
        </div>
        <button
          className="carousel-control-prev"
          type="button"
          data-bs-target="#carouselExampleCaptions"
          data-bs-slide="prev"
        >
          <span
            className="carousel-control-prev-icon"
            aria-hidden="true"
          ></span>
          <span className="visually-hidden">Previous</span>
        </button>
        <button
          className="carousel-control-next"
          type="button"
          data-bs-target="#carouselExampleCaptions"
          data-bs-slide="next"
        >
          <span
            className="carousel-control-next-icon"
            aria-hidden="true"
          ></span>
          <span className="visually-hidden">Next</span>
        </button>
      </div>

      <section id="about" className="about section-padding">
        <div className="container">
          <div className="row">
            <div className="col-lg-4 col-md-12 col-12">
              <div className="about-img">
                <img
                  src="https://img.freepik.com/free-photo/horiontal-view-round-shaped-free-space-IFY-vegetables-fallen-oil-bottle-eggs-lemons-spices-black-background_140725-160494.jpg?size=626&ext=jpg&ga=GA1.1.386372595.1698278400&semt=ais"
                  alt="..."
                  className="img-fluid"
                />
              </div>
            </div>
            <div className="col-lg-8 col-md-12 col-12 ps-lg-5 mt-md-5">
              <div className="about-text">
                <h2>
                  We Provide IFY <br />
                  Fruits and Vegetables
                </h2>
                <p>
                  <span className="text-warning">HEALTH</span>IFY is a website
                  which is developed to facilitate the efficient ordering,
                  delivery, and management of IFY fruits and vegetables. It is a
                  user-friendly interface where customers can browse the
                  available fruits and vegetables and place their orders
                  anywhere.
                </p>
                <a href="/" className="btn btn-warning">
                  Learn More
                </a>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section id="services" className="services section-padding">
        <div className="container">
          <div className="row">
            <div className="col-md-12">
              <div className="section-header text-center pb-5">
                <h2>Our Services</h2>
                <p>
                  We have enhanced high service based on delivering
                  <br />
                  IFY fruits and vegetables in a faster way
                </p>
              </div>
            </div>
          </div>
          <div className="row">
            <div className="col-12 col-md-12 col-lg-4">
              <div className="card text-white text-center bg-dark pb-2">
                <div className="card-body">
                  <i className="bi bi-subtract"></i>
                  <h3 className="card-title">Fast delivery</h3>
                  <p className="lead">
                    We have delivering IFY fruits and vegetables to the
                    customers which have been harvested from the farmers. The
                    faster mode of transportation in delivery is enhanced in our
                    website.
                  </p>
                  <button className="btn btn-warning text-dark">
                    Read More
                  </button>
                </div>
              </div>
            </div>
            <div className="col-12 col-md-12 col-lg-4">
              <div className="card text-white text-center bg-dark pb-2">
                <div className="card-body">
                  <i className="bi bi-slack"></i>
                  <h3 className="card-title">IFY and Clean</h3>
                  <p className="lead">
                    The fruits and vegetables are IFY and clean which is
                    harvested from the farmers and within a short span, we
                    collect the IFY items in our storage in our website.
                  </p>
                  <button className="btn btn-warning text-dark">
                    Read More
                  </button>
                </div>
              </div>
            </div>
            <div className="col-12 col-md-12 col-lg-4">
              <div className="card text-white text-center bg-dark pb-2">
                <div className="card-body">
                  <i className="bi bi-watch"></i>
                  <h3 className="card-title">Save Time</h3>
                  <p className="lead">
                    Ordering in our website which saves time instead of direct
                    way to market as we deliver the highly IFY fruits and
                    vegetables in a faster mode of delivery access.
                  </p>
                  <button className="btn btn-warning text-dark">
                    Read More
                  </button>
                </div>
              </div>
            </div>

            <div className="row"></div>
          </div>
        </div>
      </section>

      <section id="portfolio" className="portfolio section-padding">
        <div className="container">
          <div className="row">
            <div className="col-md-12">
              <div className="section-header text-center pb-5">
                <h2>Our Achievements</h2>
                <p>
                  Enhanced many farmers economic welfare
                  <br />
                  Awarded Second highest grossing website in Asia level
                  <br />
                  Assured economic profits for farmers and customers
                </p>
              </div>
            </div>
          </div>

          <div className="row">
            <div className="col-12 col-md-12 col-lg-4">
              <div className="card text-center bg-white pb-2">
                <div className="card-body text-dark">
                  <div className="img-area mb-4">
                    <img
                      src="https://e0.pxfuel.com/wallpapers/908/206/desktop-wallpaper-social-media-social-network.jpg"
                      alt="..."
                      class="img-fluid"
                    />
                  </div>
                  <h3 classcard-title>Social Network</h3>
                  <p className="lead">
                    We have enhanced the connecting different locations to
                    deliver the fruits and vegetables from the farm with the
                    help of our engineers.
                  </p>
                  <button className="btn bg-warning text-dark">
                    Learn More
                  </button>
                </div>
              </div>
            </div>
            <div className="col-12 col-md-12 col-lg-4">
              <div className="card text-center bg-white pb-2">
                <div className="card-body text-dark">
                  <div className="img-area mb-4">
                    <img
                      src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSq16oDVd10-leO6_IJvHriOOyYw8reU6-GlQ&usqp=CAU"
                      alt="..."
                      class="img-fluid"
                    />
                  </div>
                  <h3 classcard-title>Harvesting technology</h3>
                  <p className="lead">
                    The fruits and vegetables have been cultivated and harvested
                    from the farmers with the proper and controllable way of
                    planting the seeds.
                  </p>
                  <button className="btn bg-warning text-dark">
                    Learn More
                  </button>
                </div>
              </div>
            </div>
            <div className="col-12 col-md-12 col-lg-4">
              <div className="card text-center bg-white pb-2">
                <div className="card-body text-dark">
                  <div className="img-area mb-4">
                    <img
                      src="https://png.pngitem.com/pimgs/s/22-222412_home-delivery-hd-png-download.png"
                      alt="..."
                      class="img-fluid"
                    />
                  </div>
                  <h3 classcard-title>Speed delivery</h3>
                  <p className="lead">
                    The fruits and vegetables are delivered to your hands in
                    faster mode with a safety protocols and a well-rated
                    delivery partners.
                  </p>
                  <button className="btn bg-warning text-dark">
                    Learn More
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section id="team" className="team section-padding">
        <div className="container">
          <div className="row">
            <div className="col-md-12">
              <div className="section-header text-center pb-5">
                <h2>Team Coordinates</h2>
                <p>
                  Our team members contributed to develop the{" "}
                  <span className="text-warning">HEALTH</span>IFY website and
                  unleashing in the current world
                </p>
              </div>
            </div>
          </div>
          <div className="row">
            <div className="col-12 col-md-6 col-lg-3">
              <div className="card text-center">
                <div className="card-body">
                  <img
                    src="/images/dhiyanesh.jpg"
                    alt="..."
                    className="img-fluid rounded-circle"
                  />
                  <h5 className="card-title py-2">Dhiyanesh</h5>
                  <p className="card-text">Help-line networking developer</p>

                  <p className="socials">
                    <i className="bi bi-twitter text-dark mx-1"></i>
                    <i className="bi bi-facebook text-dark mx-1"></i>
                    <i className="bi bi-linkedin text-dark mx-1"></i>
                    <i className="bi bi-instagram text-dark mx-1"></i>
                  </p>
                </div>
              </div>
            </div>
            <div className="col-12 col-md-6 col-lg-3">
              <div className="card text-center">
                <div className="card-body">
                  <img
                    src="/images/avi.jpg"
                    alt="..."
                    className="img-fluid rounded-circle"
                  />
                  <h5 className="card-title py-2">Avinash Kumar</h5>
                  <p className="card-text">Front-end developer</p>

                  <p className="socials">
                    <i className="bi bi-twitter text-dark mx-1"></i>
                    <i className="bi bi-facebook text-dark mx-1"></i>
                    <i className="bi bi-linkedin text-dark mx-1"></i>
                    <i className="bi bi-instagram text-dark mx-1"></i>
                  </p>
                </div>
              </div>
            </div>
            <div className="col-12 col-md-6 col-lg-3">
              <div className="card text-center">
                <div className="card-body">
                  <img
                    src="/images/dinesh.jpg"
                    alt="..."
                    className="img-fluid rounded-circle"
                  />
                  <h5 className="card-title py-2">Dinesh</h5>
                  <p className="card-text">Image detector data analyst</p>

                  <p className="socials">
                    <i className="bi bi-twitter text-dark mx-1"></i>
                    <i className="bi bi-facebook text-dark mx-1"></i>
                    <i className="bi bi-linkedin text-dark mx-1"></i>
                    <i className="bi bi-instagram text-dark mx-1"></i>
                  </p>
                </div>
              </div>
            </div>
            <div className="col-12 col-md-6 col-lg-3">
              <div className="card text-center">
                <div className="card-body">
                  <img
                    src="/images/sai.jpg"
                    alt="..."
                    className="img-fluid rounded-circle"
                  />
                  <h5 className="card-title py-2">Sai Nikhil</h5>
                  <p className="card-text">Back-end developer</p>

                  <p className="socials">
                    <i className="bi bi-twitter text-dark mx-1"></i>
                    <i className="bi bi-facebook text-dark mx-1"></i>
                    <i className="bi bi-linkedin text-dark mx-1"></i>
                    <i className="bi bi-instagram text-dark mx-1"></i>
                  </p>
                </div>
              </div>
            </div>
            <div className="col-12 col-md-6 col-lg-3">
              <div className="card text-center">
                <div className="card-body">
                  <img
                    src="/images/deshik.jpg"
                    alt="..."
                    className="img-fluid rounded-circle"
                  />
                  <h5 className="card-title py-2">Deshik</h5>
                  <p className="card-text">Path-track networking developer</p>

                  <p className="socials">
                    <i className="bi bi-twitter text-dark mx-1"></i>
                    <i className="bi bi-facebook text-dark mx-1"></i>
                    <i className="bi bi-linkedin text-dark mx-1"></i>
                    <i className="bi bi-instagram text-dark mx-1"></i>
                  </p>
                </div>
              </div>
            </div>
          </div>
        </div>
      </section>

      <section id="contact" className="contact section-padding">
        <div className="container">
          <div className="row">
            <div className="cpol-md-12">
              <div className="section-header text-center pb-5">
                <h2>Contact us</h2>
                <p>Contact us, if you have any queries</p>
              </div>
            </div>
          </div>
          <div className="row m-0">
            <div className="col-md-12 p-0 pt-4 pb-4">
              <form action="/" className="bg-light p-4.m-auto">
                <div className="row">
                  <div className="col-md-12">
                    <div className="mb-3">
                      <input
                        type="text"
                        className="form-control"
                        required
                        placeholder="Your Full name"
                      />
                    </div>
                  </div>
                  <div className="col-md-12">
                    <div className="mb-3">
                      <input
                        type="email"
                        className="form-control"
                        required
                        placeholder="Your Email here"
                      />
                    </div>
                  </div>
                  <div className="col-md-12">
                    <div className="mb-3">
                      <textarea
                        rows="3"
                        required
                        className="form-control"
                        placeholder="Your Query Here"
                      />
                    </div>
                  </div>
                  <button className="btn btn-warning btn-md btn-block mt-3">
                    Send Now
                  </button>
                </div>
              </form>
            </div>
          </div>
        </div>
      </section>

      <footer className="bg-dark p-1 text-center">
        <div className="container">
          <p className="text-white">
            All Right Reserved <span className="text-warning">HEALTH</span>IFY
          </p>
        </div>
      </footer>
    </>
  );
}
