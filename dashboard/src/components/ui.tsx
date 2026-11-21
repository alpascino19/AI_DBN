import type { ReactNode } from 'react'

export function Card({ title, children, className = '' }: { title: string; children: ReactNode; className?: string }) {
  return (
    <section className={`rounded-xl border border-slate-800 bg-slate-900 p-4 ${className}`}>
      <h2 className="mb-3 text-xs font-semibold uppercase tracking-wider text-slate-400">{title}</h2>
      {children}
    </section>
  )
}
