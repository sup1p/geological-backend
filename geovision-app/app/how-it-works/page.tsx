import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Upload, Cpu, Download, Sparkles } from "lucide-react"

const steps = [
  {
    icon: Upload,
    title: "Upload Your Data",
    description:
      "Upload geological maps, legends, and stratigraphic columns. Our platform supports multiple image formats.",
  },
  {
    icon: Sparkles,
    title: "AI Analysis",
    description:
      "Our advanced AI analyzes your geological data, identifying layers, formations, and patterns with high accuracy.",
  },
  {
    icon: Cpu,
    title: "Generate Sections",
    description:
      "Create enhanced geological cross-sections with detailed layer information and accurate depth representations.",
  },
  {
    icon: Download,
    title: "Export Results",
    description:
      "Download high-resolution images and detailed reports of your geological sections for further analysis.",
  },
]

export default function HowItWorksPage() {
  return (
    <main className="container px-4 py-12">
      <div className="max-w-4xl mx-auto space-y-12">
        <div className="text-center space-y-4">
          <h1 className="text-4xl font-bold text-balance">How GeoVision Works</h1>
          <p className="text-lg text-muted-foreground max-w-2xl mx-auto">
            Transform your geological data into actionable insights in four simple steps.
          </p>
        </div>

        <div className="space-y-8">
          {steps.map((step, index) => (
            <Card key={step.title}>
              <CardHeader>
                <div className="flex items-start gap-4">
                  <div className="flex h-12 w-12 shrink-0 items-center justify-center rounded-lg bg-primary text-primary-foreground">
                    <step.icon className="h-6 w-6" />
                  </div>
                  <div className="space-y-1">
                    <div className="flex items-center gap-3">
                      <span className="text-sm font-medium text-muted-foreground">Step {index + 1}</span>
                    </div>
                    <CardTitle>{step.title}</CardTitle>
                    <CardDescription className="text-base">{step.description}</CardDescription>
                  </div>
                </div>
              </CardHeader>
            </Card>
          ))}
        </div>

        <Card className="bg-muted">
          <CardHeader>
            <CardTitle>Advanced Features</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid gap-4 md:grid-cols-2">
              <div>
                <h3 className="font-semibold mb-2">Real-time AI Assistant</h3>
                <p className="text-sm text-muted-foreground">
                  Chat with Strato, our AI assistant, for instant help with geological analysis and interpretation.
                </p>
              </div>
              <div>
                <h3 className="font-semibold mb-2">Multiple Data Types</h3>
                <p className="text-sm text-muted-foreground">
                  Support for magnetic, gravity, seismic, and electrical survey data integration.
                </p>
              </div>
              <div>
                <h3 className="font-semibold mb-2">Anomaly Detection</h3>
                <p className="text-sm text-muted-foreground">
                  Automatically identify geological anomalies and areas of interest in your data.
                </p>
              </div>
              <div>
                <h3 className="font-semibold mb-2">Collaborative Tools</h3>
                <p className="text-sm text-muted-foreground">
                  Share projects with team members and collaborate in real-time on geological analyses.
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </main>
  )
}
