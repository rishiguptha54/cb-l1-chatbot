import type { LucideIcon } from "lucide-react";
import { Building2, Wrench } from "lucide-react";

/** App-wide constants. Adjust `JIRA_BROWSE_BASE` if your Jira host differs. */

export const APP_NAME = "CB L1 Support Chatbot";
export const APP_TAGLINE = "Unravel your defect history";
export const APP_VERSION = "1.0.0";

/**
 * Base URL used to turn a Jira key (e.g. HCBS-95506) into a clickable link.
 * Mirrors the backend `JIRA_BASE_URL` default from `config.py`.
 */
export const JIRA_BROWSE_BASE = "https://honeywell.atlassian.net/browse/";

export interface SuggestedCategory {
  icon: LucideIcon;
  title: string;
  questions: string[];
}

export const SUGGESTED_CATEGORIES: SuggestedCategory[] = [
  {
    icon: Wrench,
    title: "Diagnose & fix",
    questions: [
      "How to fix an EOM publish failure?",
      "Why does the asset sync keep failing?",
      "Steps to resolve a mapping error",
    ],
  },
  {
    icon: Building2,
    title: "Field symptoms",
    questions: [
      "Data is not coming in RBM after model publish",
      "Comfort Score not coming with value in unified app",
      "Trend not displaying in the RBM for the point",
    ],
  },
];
