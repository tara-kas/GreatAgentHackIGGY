# Agent Identification

## Overview

This directory contains resources and documentation for identifying the technical implementations of each agent in the system.

## Goal

**Identify the technical stack for each agent:**

- **Framework**: Determine which framework is used (e.g., LangGraph, CrewAI, AutoGen, etc.)
- **Model**: Identify which models power each agent
- **Architecture Pattern**: Understand what architecture patterns they follow
- **Mapping**: Create a mapping of animals to their technical implementations

## Agents

The following agents are available for testing:

| Animal | Emoji | Endpoint |
|--------|-------|----------|
| Elephant | 🐘 | `/api/elephant` |
| Fox | 🦊 | `/api/fox` |
| Eagle | 🦅 | `/api/eagle` |
| Ant | 🐜 | `/api/ant` |
| Wolf | 🐺 | `/api/wolf` |
| Bear | 🐻 | `/api/bear` |
| Chameleon | 🦎 | `/api/chameleon` |

## Identification Strategy

Each agent has unique characteristics that can be discovered through testing. Use the following signals to identify them:

- **Response Times**: Different frameworks/models have varying response characteristics
- **Behavior Patterns**: Observe how each agent processes and responds to requests
- **Error Messages**: Framework-specific error messages can reveal the underlying technology
- **Other Observable Signals**: Any other distinguishing features in the API responses

## Bonus Points

Correctly identifying agent types (framework, model, architecture pattern) earns bonus points in judging!
