import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Brain, Layers, Zap, Shield, Globe, BarChart } from "lucide-react"

const features = [
  {
    icon: Brain,
    title: "AI-Powered Analysis",
    description: "Advanced machine learning algorithms analyze geological data with unprecedented accuracy and speed.",
  },
  {
    icon: Layers,
    title: "Multi-Layer Visualization",
    description:
      "Create detailed cross-sections showing multiple geological layers with accurate depth and composition data.",
  },
  {
    icon: Zap,
    title: "Real-Time Processing",
    description: "Get instant results with our optimized processing pipeline that handles large datasets efficiently.",
  },
  {
    icon: Shield,
    title: "Secure & Private",
    description: "Your geological data is encrypted and stored securely with enterprise-grade security measures.",
  },
  {
    icon: Globe,
    title: "Global Coverage",
    description: "Work with geological data from anywhere in the world with support for multiple coordinate systems.",
  },
  {
    icon: BarChart,
    title: "Advanced Analytics",
    description: "Comprehensive statistical analysis and reporting tools for in-depth geological insights.",
  },
]

export default function FeaturesPage() {
  return (
    <main className="container px-4 py-12">
      <div className="max-w-6xl mx-auto space-y-12">
        <div className="text-center space-y-4">
          <h1 className="text-4xl font-bold text-balance">Powerful Features for Geological Analysis</h1>
          <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
            Everything you need to analyze, visualize, and understand geological data in one comprehensive platform.
          </p>
        </div>

        <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
          {features.map((feature) => (
            <Card key={feature.title} className="hover:shadow-lg transition-shadow">
              <CardHeader>
                <div className="flex h-12 w-12 items-center justify-center rounded-lg bg-primary/10 mb-4">
                  <feature.icon className="h-6 w-6 text-primary" />
                </div>
                <CardTitle>{feature.title}</CardTitle>
                <CardDescription className="text-base">{feature.description}</CardDescription>
              </CardHeader>
            </Card>
          ))}
        </div>

        <Card className="bg-primary text-primary-foreground">
          <CardHeader>
            <CardTitle className="text-2xl">Ready to get started?</CardTitle>
            <CardDescription className="text-primary-foreground/80 text-base">
              Join thousands of geologists using GeoVision to analyze geological data more efficiently.
            </CardDescription>
          </CardHeader>
          <CardContent>
            <div className="flex flex-wrap gap-4">
              <button className="rounded-md bg-primary-foreground px-6 py-2 text-sm font-medium text-primary hover:bg-primary-foreground/90 transition-colors">
                Start Free Trial
              </button>
              <button className="rounded-md border border-primary-foreground/20 px-6 py-2 text-sm font-medium text-primary-foreground hover:bg-primary-foreground/10 transition-colors">
                Schedule Demo
              </button>
            </div>
          </CardContent>
        </Card>
      </div>
    </main>
  )
}
