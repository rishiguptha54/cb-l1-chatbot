import { motion } from "framer-motion";
import { LandingNav } from "@/components/landing/LandingNav";
import { Hero } from "@/components/landing/Hero";
import { Features } from "@/components/landing/Features";
import { HowItWorks } from "@/components/landing/HowItWorks";
import { WhyChoose } from "@/components/landing/WhyChoose";
import { Footer } from "@/components/landing/Footer";

export default function LandingPage() {
  return (
    <motion.div
      initial={{ opacity: 0 }}
      animate={{ opacity: 1 }}
      exit={{ opacity: 0 }}
      transition={{ duration: 0.3 }}
      className="min-h-full bg-background"
    >
      <LandingNav />
      <main>
        <Hero />
        <Features />
        <HowItWorks />
        <WhyChoose />
      </main>
      <Footer />
    </motion.div>
  );
}
