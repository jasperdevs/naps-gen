import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  devIndicators: false,
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: "http://127.0.0.1:8001/api/:path*",
      },
      {
        source: "/ws",
        destination: "http://127.0.0.1:8001/ws",
      },
      {
        source: "/static/:path*",
        destination: "http://127.0.0.1:8001/static/:path*",
      },
      {
        source: "/health",
        destination: "http://127.0.0.1:8001/health",
      },
    ];
  },
};

export default nextConfig;
