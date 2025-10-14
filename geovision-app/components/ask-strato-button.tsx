"use client"

import { Plus } from "lucide-react"
import { Button } from "@/components/ui/button"
import { cn } from "@/lib/utils"

interface AskStratoButtonProps {
  onClick: () => void
  disabled?: boolean
  dimmed?: boolean
}

export function AskStratoButton({ onClick, disabled = false, dimmed = false }: AskStratoButtonProps) {
  return (
    <Button
      onClick={disabled ? undefined : onClick}
      size="lg"
      disabled={disabled}
      className={cn(
        "fixed bottom-6 right-6 z-30 h-14 rounded-full px-6 shadow-lg hover:shadow-xl transition-all duration-300",
        dimmed && "opacity-30 pointer-events-none",
        disabled && "cursor-not-allowed"
      )}
    >
      <span className="mr-2">Ask Strato</span>
      <Plus className="h-5 w-5" />
    </Button>
  )
}
