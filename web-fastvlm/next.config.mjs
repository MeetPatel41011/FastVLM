/** @type {import('next').NextConfig} */
const nextConfig = {
  turbopack: {},
  output: process.env.STATIC_EXPORT ? 'export' : undefined,
  // Only use rewrites for local development; static export handles routing natively on Modal
  ...(process.env.STATIC_EXPORT ? {} : {
    async rewrites() {
      return [
        {
          source: '/api/backend/:path*',
          destination: 'http://127.0.0.1:8000/:path*',
        },
      ]
    }
  })
};

export default nextConfig;