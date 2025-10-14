"use client"

import { useEffect, useState } from "react"
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card"
import { Button } from "@/components/ui/button"
import { getUserSections, deleteUserSection, type UserSection } from "@/lib/api"
import { useToast } from "@/hooks/use-toast"
import { Trash2, ExternalLink } from "lucide-react"

export default function ProjectsPage() {
  const [sections, setSections] = useState<UserSection[]>([])
  const [isLoading, setIsLoading] = useState(true)
  const [page, setPage] = useState(1)
  const [totalPages, setTotalPages] = useState(1)
  const { toast } = useToast()

  useEffect(() => {
    loadSections()
  }, [page])

  const loadSections = async () => {
    try {
      setIsLoading(true)
      const data = await getUserSections(page, 10)
      setSections(data.items)
      setTotalPages(data.total_pages)
    } catch (error) {
      console.error("[v0] Failed to load sections:", error)
      toast({
        title: "Error",
        description: "Failed to load sections. Please log in.",
        variant: "destructive",
      })
    } finally {
      setIsLoading(false)
    }
  }

  const handleDelete = async (sectionId: number) => {
    if (!confirm("Are you sure you want to delete this section?")) return

    try {
      await deleteUserSection(sectionId)
      toast({
        title: "Success",
        description: "Section deleted successfully",
      })
      loadSections()
    } catch (error) {
      toast({
        title: "Error",
        description: "Failed to delete section",
        variant: "destructive",
      })
    }
  }

  return (
    <main className="container px-4 py-12">
      <div className="max-w-6xl mx-auto space-y-8">
        <div className="space-y-4">
          <h1 className="text-4xl font-bold text-balance">Your Projects</h1>
          <p className="text-lg text-muted-foreground">
            Manage and view all your geological analysis projects in one place.
          </p>
        </div>

        {isLoading ? (
          <Card>
            <CardContent className="p-12 text-center">
              <p className="text-muted-foreground">Loading sections...</p>
            </CardContent>
          </Card>
        ) : sections.length === 0 ? (
          <Card>
            <CardHeader>
              <CardTitle>No Sections Yet</CardTitle>
              <CardDescription>Upload a map to create your first geological section</CardDescription>
            </CardHeader>
            <CardContent>
              <Button asChild>
                <a href="/">Upload Map</a>
              </Button>
            </CardContent>
          </Card>
        ) : (
          <>
            <div className="grid gap-6 md:grid-cols-2 lg:grid-cols-3">
              {sections.map((section) => (
                <Card key={section.id}>
                  <CardHeader>
                    <CardTitle className="text-base">Section #{section.id}</CardTitle>
                    <CardDescription>{new Date(section.created_at).toLocaleDateString()}</CardDescription>
                  </CardHeader>
                  <CardContent className="space-y-4">
                    <div className="aspect-video bg-muted rounded-md overflow-hidden">
                      <img
                        src={section.section_url || "/placeholder.svg"}
                        alt={`Section ${section.id}`}
                        className="w-full h-full object-cover"
                      />
                    </div>
                    <div className="flex gap-2">
                      <Button variant="outline" size="sm" className="flex-1 bg-transparent" asChild>
                        <a href={section.section_url} target="_blank" rel="noopener noreferrer">
                          <ExternalLink className="h-4 w-4 mr-2" />
                          View
                        </a>
                      </Button>
                      <Button variant="destructive" size="sm" onClick={() => handleDelete(section.id)}>
                        <Trash2 className="h-4 w-4" />
                      </Button>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>

            {totalPages > 1 && (
              <div className="flex justify-center gap-2">
                <Button variant="outline" disabled={page === 1} onClick={() => setPage(page - 1)}>
                  Previous
                </Button>
                <span className="flex items-center px-4 text-sm text-muted-foreground">
                  Page {page} of {totalPages}
                </span>
                <Button variant="outline" disabled={page === totalPages} onClick={() => setPage(page + 1)}>
                  Next
                </Button>
              </div>
            )}
          </>
        )}
      </div>
    </main>
  )
}
