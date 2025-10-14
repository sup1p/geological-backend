"use client"

import { Menu, X } from "lucide-react"
import { cn } from "@/lib/utils"
import { useRouter } from "next/navigation"
import { useAuth } from "@/contexts/auth-context"

interface SidebarProps {
  isOpen: boolean
  onClose: () => void
  onOpenUploadModal?: () => void
}

const tools = [
  { name: "AI Analysis", enabled: true, requiresAuth: true },
  { name: "Add Magnetic Data", enabled: false, requiresAuth: false },
  { name: "Add Gravity Data", enabled: false, requiresAuth: false },
  { name: "Seismic Upload", enabled: false, requiresAuth: false },
  { name: "Electrical Survey", enabled: false, requiresAuth: false },
  { name: "Anomaly Detector", enabled: false, requiresAuth: false },
  { name: "Export Results", enabled: false, requiresAuth: false },
]

export function Sidebar({ isOpen, onClose, onOpenUploadModal }: SidebarProps) {
  const router = useRouter()
  const { isAuthenticated } = useAuth()

  const handleToolClick = (tool: (typeof tools)[0]) => {
    if (!tool.enabled) return

    if (tool.requiresAuth) {
      if (!isAuthenticated) {
        router.push("/login")
        onClose()
      } else {
        // If authenticated and it's AI Analysis, open the upload modal
        if (tool.name === "AI Analysis" && onOpenUploadModal) {
          onOpenUploadModal()
          onClose()
        }
      }
    }
  }

  const handleAboutClick = () => {
    router.push("/about")
    onClose()
  }

  return (
    <>
      {isOpen && <div className="fixed inset-0 z-40 bg-black/30 backdrop-blur-sm" onClick={onClose} />}

      <aside
        className={cn(
          "fixed left-0 top-0 z-50 h-full w-72 bg-card border-r border-border shadow-2xl transition-transform duration-300 ease-in-out",
          isOpen ? "translate-x-0" : "-translate-x-full",
        )}
      >
        <div className="flex h-full flex-col">
          <div className="flex items-center justify-between border-b border-border bg-muted/30 px-6 py-4">
            <div className="flex items-center gap-3">
              <Menu className="h-5 w-5 text-primary" />
              <h2 className="text-lg font-semibold text-foreground">Tools</h2>
            </div>
            <button
              onClick={onClose}
              className="rounded-md p-1 hover:bg-muted transition-colors"
              aria-label="Close sidebar"
            >
              <X className="h-5 w-5 text-muted-foreground" />
            </button>
          </div>

          <nav className="flex-1 overflow-y-auto py-2">
            {tools.map((tool, index) => (
              <button
                key={tool.name}
                disabled={!tool.enabled}
                onClick={() => handleToolClick(tool)}
                className={cn(
                  "w-full px-6 py-3.5 text-left text-sm font-medium transition-all border-b border-border/50",
                  tool.enabled
                    ? "text-foreground hover:bg-accent hover:text-accent-foreground cursor-pointer"
                    : "text-muted-foreground/40 cursor-not-allowed bg-muted/20",
                  index === 0 && tool.enabled && "bg-accent/50",
                )}
              >
                {tool.name}
              </button>
            ))}
          </nav>

          <div className="border-t border-border bg-muted/30">
            <button
              onClick={handleAboutClick}
              className="w-full px-6 py-4 text-left text-sm font-medium text-foreground hover:bg-accent hover:text-accent-foreground transition-colors"
            >
              About us
            </button>
          </div>
        </div>
      </aside>
    </>
  )
}
