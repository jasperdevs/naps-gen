"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";
import { cn } from "@/lib/utils";
import { useWebSocket } from "@/hooks/use-websocket";
import { Sparkles, GraduationCap, Database, Settings, Wifi, WifiOff, ServerOff } from "lucide-react";

export const NAV_ITEMS = [
  { href: "/generate", label: "Generate", icon: Sparkles },
  { href: "/training", label: "Training", icon: GraduationCap },
  { href: "/dataset", label: "Dataset", icon: Database },
  { href: "/settings", label: "Settings", icon: Settings },
];

export function Sidebar() {
  const pathname = usePathname();
  const { connected, backendOnline } = useWebSocket();

  return (
    <aside className="sticky top-0 flex h-screen w-52 shrink-0 flex-col border-r border-border bg-sidebar">
      <div className="flex items-center gap-2 border-b border-border px-4 py-4">
        <img src="/logo.png" alt="NapsGen" className="h-6 w-6 rounded-md object-cover" />
        <span className="text-sm font-semibold">NapsGen</span>
      </div>
      <nav className="flex-1 space-y-0.5 px-2 py-3">
        {NAV_ITEMS.map(({ href, label, icon: Icon }) => (
          <Link key={href} href={href} className={cn(
            "flex items-center gap-2.5 rounded-md px-2.5 py-2 text-sm transition-colors",
            pathname === href ? "bg-accent text-accent-foreground font-medium" : "text-muted-foreground hover:bg-accent/50 hover:text-foreground"
          )}>
            <Icon className="h-4 w-4" />
            {label}
          </Link>
        ))}
      </nav>
      <div className="border-t border-border px-4 py-3">
        <div className="flex items-center gap-2 text-xs text-muted-foreground">
          {!backendOnline ? (
            <><ServerOff className="h-3 w-3 text-destructive" /><span>Server offline</span></>
          ) : connected ? (
            <><Wifi className="h-3 w-3 text-green-500" /><span>Live</span></>
          ) : (
            <><WifiOff className="h-3 w-3 text-yellow-500" /><span>Reconnecting</span></>
          )}
        </div>
      </div>
    </aside>
  );
}
