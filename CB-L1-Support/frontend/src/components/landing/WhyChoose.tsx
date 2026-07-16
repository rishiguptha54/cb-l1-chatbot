import { Link } from "react-router-dom";
import { motion } from "framer-motion";
import { ArrowRight, BadgeCheck, Coins, RefreshCw, ShieldCheck } from "lucide-react";
import { SectionHeading } from "./Features";

const REASONS = [
  {
    icon: BadgeCheck,
    title: "Grounded answers",
    desc: "Every answer cites the real defects and documents behind it — no hallucinated fixes.",
  },
  {
    icon: Coins,
    title: "Fewer tokens, lower cost",
    desc: "Only the top-matched evidence reaches the model, so answers stay sharp and inexpensive.",
  },
  {
    icon: RefreshCw,
    title: "Resilient by design",
    desc: "Degrades gracefully and times out safely — it always responds with something useful.",
  },
  {
    icon: ShieldCheck,
    title: "Read-only & safe",
    desc: "Jira access is strictly read-only through MCP; your source systems are never modified.",
  },
];

export function WhyChoose() {
  return (
    <section id="why-us" className="py-20 sm:py-28">
      <div className="mx-auto max-w-6xl px-4 sm:px-6">
        <SectionHeading
          eyebrow="Why choose it"
          title="Built for trust, speed, and scale"
          subtitle="Not a generic chatbot — a purpose-built layer over your defect intelligence."
        />

        <div className="mt-14 grid gap-5 md:grid-cols-2">
          {REASONS.map((r, i) => (
            <motion.div
              key={r.title}
              initial={{ opacity: 0, y: 24 }}
              whileInView={{ opacity: 1, y: 0 }}
              viewport={{ once: true, margin: "-60px" }}
              transition={{ delay: (i % 2) * 0.1, duration: 0.5 }}
              className="flex gap-4 rounded-2xl border border-border bg-card p-6 shadow-soft transition-all hover:border-primary/40 hover:shadow-elevated"
            >
              <div className="grid h-11 w-11 shrink-0 place-items-center rounded-xl bg-primary/10 text-primary ring-1 ring-primary/15">
                <r.icon className="h-5 w-5" />
              </div>
              <div>
                <h3 className="text-base font-semibold text-foreground">{r.title}</h3>
                <p className="mt-1.5 text-sm leading-relaxed text-muted-foreground">{r.desc}</p>
              </div>
            </motion.div>
          ))}
        </div>

        {/* CTA band */}
        <motion.div
          initial={{ opacity: 0, y: 24 }}
          whileInView={{ opacity: 1, y: 0 }}
          viewport={{ once: true, margin: "-60px" }}
          transition={{ duration: 0.5 }}
          className="relative mt-16 overflow-hidden rounded-3xl border border-border bg-brand-gradient bg-[length:200%_200%] p-10 text-center shadow-elevated sm:p-14"
        >
          <div className="pointer-events-none absolute inset-0 bg-grid opacity-20" aria-hidden="true" />
          <div className="relative">
            <h3 className="text-2xl font-bold tracking-tight text-white sm:text-3xl">
              Make your defect history your compass.
            </h3>
            <p className="mx-auto mt-3 max-w-xl text-sm text-white/85 sm:text-base">
              Ask any question about CB and get a fix, grounded in real history — with every source
              cited, in seconds.
            </p>
            <Link
              to="/chat"
              className="group mt-8 inline-flex h-12 items-center justify-center gap-2 rounded-xl bg-white px-8 text-sm font-semibold text-brand-700 shadow-lg transition-transform hover:scale-[1.02]"
            >
              Get Started
              <ArrowRight className="h-4 w-4 transition-transform group-hover:translate-x-1" />
            </Link>
          </div>
        </motion.div>
      </div>
    </section>
  );
}
