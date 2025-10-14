export default function AboutPage() {
  return (
    <main className="container px-4 py-12">
      <div className="max-w-4xl mx-auto space-y-8">
        <h1 className="text-4xl font-bold text-balance">About GeoVision</h1>

        <div className="prose prose-lg max-w-none">
          <p className="text-lg text-muted-foreground leading-relaxed">
            GeoVision is a cutting-edge geological analysis platform that combines advanced AI technology with
            traditional geological methods to help researchers, geologists, and earth scientists understand subsurface
            structures with unprecedented clarity.
          </p>

          <h2 className="text-2xl font-semibold mt-8 mb-4">Our Mission</h2>
          <p className="text-muted-foreground leading-relaxed">
            We aim to democratize geological analysis by making sophisticated tools accessible to professionals and
            researchers worldwide. Our platform transforms complex geological data into actionable insights.
          </p>

          <h2 className="text-2xl font-semibold mt-8 mb-4">Our Technology</h2>
          <p className="text-muted-foreground leading-relaxed">
            Powered by state-of-the-art AI and machine learning algorithms, GeoVision can analyze geological maps,
            seismic data, and various geophysical measurements to create detailed cross-sections and 3D models of
            subsurface structures.
          </p>

          <h2 className="text-2xl font-semibold mt-8 mb-4">Contact Us</h2>
          <p className="text-muted-foreground leading-relaxed">
            Have questions or want to learn more? Visit our{" "}
            <a href="/contact" className="text-primary hover:underline">
              contact page
            </a>{" "}
            to get in touch with our team.
          </p>
        </div>
      </div>
    </main>
  )
}
