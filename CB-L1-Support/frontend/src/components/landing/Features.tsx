import { motion } from "framer-motion";
import { Crosshair, History, Layers, SlidersHorizontal } from "lucide-react";

const FEATURES = [
  {
    icon: History,
    label: "Evidence & clarity",
    title: "See how defects were fixed",
    desc: "Real historical fixes for any symptom or ticket — cited by Jira key, never stale.",
  },
  {
    icon: Crosshair,
    label: "Root-cause analysis",
    title: "Pinpoint the likely cause",
    desc: "Infers the most probable root cause from the most similar past defects.",
  },
  {
    icon: SlidersHorizontal,
    label: "Precision AI & cost",
    title: "Feed AI only what matters",
    desc: "Sends only the top-matched evidence to the model — fewer tokens, sharper answers.",
  },
  {
    icon: Layers,
    label: "Dual knowledge",
    title: "Defects + docs, together",
    desc: "Fuses historical defects with product-doc search and labels every source.",
  },
];

export function Features() {
  return (
    <section id="features" className="relative py-20 sm:py-28">
      <div className="mx-auto max-w-6xl px-4 sm:px-6">
        <SectionHeading
          eyebrow="What it does"
          title="Turn your defect history into answers"
          subtitle="Reads your defect history and product docs, then answers root-cause and fix questions in plain English."
        />

        <div className="mt-14 grid gap-5 sm:grid-cols-2 lg:grid-cols-4">
          {FEATURES.map((f, i) => (
            <motion.article
              key={f.title}
              initial={{ opacity: 0, y: 24 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: "-60px" }}
              transition={{ delay: (i % 4) * 0.08, duration: 0.5 }}
              className="group relative overflow-hidden rounded-2xl border border-border bg-card p-6 shadow-soft transition-all hover:-translate-y-1 hover:border-primary/40 hover:shadow-elevated"
            >
              <div
                className="pointer-events-none absolute -right-8 -top-8 h-24 w-24 rounded-full bg-brand-gradient opacity-0 blur-2xl transition-opacity duration-300 group-hover:opacity-20"
                aria-hidden="true"
              />
              <div className="grid h-12 w-12 place-items-center rounded-xl bg-primary/10 text-primary ring-1 ring-primary/15 transition-colors group-hover:bg-primary group-hover:text-primary-foreground">
                <f.icon className="h-6 w-6" />
              </div>
              <span className="mt-5 block text-[0.7rem] font-semibold uppercase tracking-[0.14em] text-primary">
                {f.label}
              </span>
              <h3 className="mt-1.5 text-lg font-semibold text-foreground">{f.title}</h3>
              <p className="mt-2 text-sm leading-relaxed text-muted-foreground">{f.desc}</p>
            </motion.article>
          ))}
        </div>
      </div>
    </section>
  );
}

export function SectionHeading({
  eyebrow,
  title,
  subtitle,
}: {
  eyebrow: string;
  title: string;
  subtitle?: string;
}) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 16 }}
      whileInView={{ opacity: 1, y: 0 }}
      viewport={{ once: true, margin: "-60px" }}
      transition={{ duration: 0.5 }}
      className="mx-auto max-w-2xl text-center"
    >
      <span className="text-xs font-semibold uppercase tracking-[0.2em] text-primary">
        {eyebrow}
      </span>
      <h2 className="mt-3 text-3xl font-bold tracking-tight text-foreground sm:text-4xl">
        {title}
      </h2>
      {subtitle && <p className="mt-4 text-base text-muted-foreground">{subtitle}</p>}
    </motion.div>
  );
}
