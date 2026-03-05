import type { Metadata } from 'next';
import './globals.css';

export const metadata: Metadata = {
  title: '주식 AI 예측 시스템 (KRX/US)',
  description: 'AI 기반 한국/미국 주식 기술적 분석 및 예측 시스템',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="ko">
      <body className="min-h-screen">{children}</body>
    </html>
  );
}
