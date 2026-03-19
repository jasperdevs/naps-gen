import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import { TooltipProvider } from "@/components/ui/tooltip";
import { Toaster } from "@/components/ui/sonner";
import { Sidebar } from "@/components/layout/sidebar";
import { MobileNav } from "@/components/layout/mobile-nav";
import { ToastListener } from "@/components/layout/toast-listener";
import { WebSocketProvider } from "@/hooks/use-websocket";
import "./globals.css";

const geistSans = Geist({ variable: "--font-geist-sans", subsets: ["latin"] });
const geistMono = Geist_Mono({ variable: "--font-geist-mono", subsets: ["latin"] });

export const metadata: Metadata = {
  title: "NapsGen",
  description: "Custom DDPM image generator trained on Roblox data",
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className={`${geistSans.variable} ${geistMono.variable} dark h-full antialiased`}>
      <body className="h-full">
        <WebSocketProvider>
          <TooltipProvider>
            <div className="flex h-full">
              {/* Desktop sidebar */}
              <div className="hidden md:block">
                <Sidebar />
              </div>
              <main className="flex-1 min-w-0 overflow-auto pb-16 md:pb-0">
                <div className="mx-auto max-w-7xl px-4 py-5 sm:px-6 lg:px-8">{children}</div>
              </main>
            </div>
            {/* Mobile bottom nav */}
            <MobileNav />
          </TooltipProvider>
          <ToastListener />
          <Toaster />
        </WebSocketProvider>
      </body>
    </html>
  );
}
