/** @type {import('next').NextConfig} */
const nextConfig = {
  // Vercel 배포 시 API는 /api/* 로 Python Serverless Function에 라우팅됨
  // rewrites: 로컬 개발 시 Python API 서버(포트 8000)로 프록시
  async rewrites() {
    return process.env.NODE_ENV === 'development'
      ? [
          {
            source: '/api/:path*',
            destination: 'http://localhost:8000/api/:path*',
          },
        ]
      : [];
  },
};

module.exports = nextConfig;
