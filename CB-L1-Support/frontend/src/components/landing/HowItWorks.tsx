import { motion } from "framer-motion";
import { Lightbulb, MessageCircleQuestion, Sparkles } from "lucide-react";
import { SectionHeading } from "./Features";

const STEPS = [
  {
    icon: MessageCircleQuestion,
    step: "01",
    title: "Ask",
    desc: "Describe a symptom, paste a Jira key, or ask an analytics question in plain language.",
  },
  {
    icon: Lightbulb,
    step: "02",
    title: "Analyze",
    desc: "Hybrid search (FAISS + BM25) ranks the most similar past defects and pulls the right docs.",
  },
  {
    icon: Sparkles,
    step: "03",
    title: "Respond",
    desc: "You get a root-cause and fix answer, cited by Jira key, with comparable defects.",
  },
];

export function HowItWorks() {
  return (
    <section id="how-it-works" className="relative overflow-hidden py-20 sm:py-28">
      <div className="pointer-events-none absolute inset-0 -z-10 bg-muted/30" aria-hidden="true" />
      <div className="mx-auto max-w-6xl px-4 sm:px-6">
        <SectionHeading
          eyebrow="How it works"
          title="From question to answer in three steps"
          subtitle="A transparent pipeline that always shows the evidence behind every response."
        />

        <div className="relative mt-16">
          {/* Connector line */}
          <div
            className="absolute left-0 right-0 top-8 hidden h-px bg-gradient-to-r from-transparent via-border to-transparent lg:block"
            aria-hidden="true"
          />
          <div className="grid gap-8 lg:grid-cols-3">
            {STEPS.map((s, i) => (
              <motion.div
                key={s.step}
                initial={{ opacity: 0, y: 24 }}
                whileInView={{ opacity: 1, y: 0 }}
                viewport={{ once: true, margin: "-60px" }}
                transition={{ delay: i * 0.12, duration: 0.5 }}
                className="relative text-center"
              >
                <div className="relative z-10 mx-auto grid h-16 w-16 place-items-center rounded-2xl border border-border bg-card text-primary shadow-soft">
                  <s.icon className="h-7 w-7" />
                  <span className="absolute -right-2 -top-2 grid h-6 w-6 place-items-center rounded-full bg-brand-gradient text-[0.65rem] font-bold text-white shadow-sm">
                    {s.step}
                  </span>
                </div>
                <h3 className="mt-6 text-xl font-semibold text-foreground">{s.title}</h3>
                <p className="mx-auto mt-2 max-w-xs text-sm leading-relaxed text-muted-foreground">
                  {s.desc}
                </p>
              </motion.div>
            ))}
          </div>
        </div>
      </div>
    </section>
  );
}
