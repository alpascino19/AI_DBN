import type { ReactNode } from 'react'

export function Card({ title, children, className = '' }: { title: string; children: ReactNode; className?: string }) {
  return (
    <section className={`rounded-xl border border-slate-800 bg-slate-900 p-4 ${className}`}>
      <h2 className="mb-3 text-sm font-semibold text-slate-300">{title}</h2>
      {children}
    </section>
  )
}
