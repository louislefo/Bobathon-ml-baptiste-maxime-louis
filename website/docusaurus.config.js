// @ts-check
import {themes as prismThemes} from 'prism-react-renderer';

const baseUrl = process.env.BASE_URL || '/';

/** @type {import('@docusaurus/types').Config} */
const config = {
  title: 'Bobathon ESILV',
  tagline: 'Evaluation Motrice et Correction des Biais — Maladie de Parkinson',
  favicon: 'img/favicon.svg',

  url: 'https://louislefo.github.io',
  baseUrl: baseUrl,

  organizationName: 'louislefo',
  projectName: 'Bobathon-ml-baptiste-maxime-louis',
  trailingSlash: false,

  onBrokenLinks: 'warn',

  i18n: {
    defaultLocale: 'fr',
    locales: ['fr'],
  },

  presets: [
    [
      'classic',
      /** @type {import('@docusaurus/preset-classic').Options} */
      ({
        docs: {
          routeBasePath: 'doc',
          sidebarPath: './sidebars.js',
        },
        blog: false,
        theme: {
          customCss: './src/css/custom.css',
        },
      }),
    ],
  ],

  themeConfig:
    /** @type {import('@docusaurus/preset-classic').ThemeConfig} */
    ({
      colorMode: {
        defaultMode: 'light',
        disableSwitch: false,
        respectPrefersColorScheme: false,
      },
      navbar: {
        title: 'Bobathon ESILV',
        logo: {
          alt: 'Probabl Logo',
          src: 'figures/probabl-logo.svg',
        },
        items: [
          {
            type: 'docSidebar',
            sidebarId: 'docsSidebar',
            position: 'left',
            label: 'Documentation',
          },
          {
            href: 'pathname:///reports/eda_visits_train.html',
            label: 'Rapport Train (EDA)',
            position: 'left',
          },
          {
            href: 'pathname:///reports/eda_X_test.html',
            label: 'Rapport Test (EDA)',
            position: 'left',
          },
          {
            href: 'https://www.kaggle.com/competitions/bobathon-esilv/',
            label: 'Competition Kaggle',
            position: 'right',
          },
          {
            href: 'https://github.com/louislefo/Bobathon-ml-baptiste-maxime-louis',
            label: 'GitHub',
            position: 'right',
          },
        ],
      },
      footer: {
        style: 'dark',
        links: [
          {
            title: 'Documentation',
            items: [
              {
                label: 'Contexte Medical',
                to: '/doc/introduction',
              },
              {
                label: 'Analyse des Donnees',
                to: '/doc/analyse-donnees-eda',
              },
              {
                label: 'Modeles et Benchmark',
                to: '/doc/modeles-et-experimentations',
              },
              {
                label: 'Guide de Reproduction',
                to: '/doc/guide-de-reproduction',
              },
            ],
          },
          {
            title: 'Rapports Interactifs',
            items: [
              {
                label: 'Exploration Train (44 590 visites)',
                href: 'pathname:///reports/eda_visits_train.html',
              },
              {
                label: 'Exploration Test (11 013 visites)',
                href: 'pathname:///reports/eda_X_test.html',
              },
            ],
          },
          {
            title: 'Liens Officiels',
            items: [
              {
                label: 'Challenge Kaggle',
                href: 'https://www.kaggle.com/competitions/bobathon-esilv/',
              },
              {
                label: 'Plateforme Skore Hub',
                href: 'https://skore.probabl.ai',
              },
              {
                label: 'Depot GitHub',
                href: 'https://github.com/louislefo/Bobathon-ml-baptiste-maxime-louis',
              },
            ],
          },
        ],
        copyright: `Projet Hackathon ML - Baptiste, Maxime, Louis | ESILV x Probabl x IBM`,
      },
      prism: {
        theme: prismThemes.github,
        darkTheme: prismThemes.dracula,
      },
    }),
};

export default config;
