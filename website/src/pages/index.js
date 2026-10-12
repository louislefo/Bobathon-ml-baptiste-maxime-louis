import React from 'react';
import Link from '@docusaurus/Link';
import useDocusaurusContext from '@docusaurus/useDocusaurusContext';
import useBaseUrl from '@docusaurus/useBaseUrl';
import Layout from '@theme/Layout';
import Heading from '@theme/Heading';
import styles from './index.module.css';

function HomepageHero() {
  const {siteConfig} = useDocusaurusContext();
  const probablLogo = useBaseUrl('/figures/probabl-logo.svg');
  const ibmLogo = useBaseUrl('/figures/ibm-logo.svg');
  const icmLogo = useBaseUrl('/figures/logo_icm.png');

  return (
    <header className={styles.heroBanner}>
      <div className="container">
        <div className={styles.brandLogos}>
          <img src={probablLogo} alt="Probabl" className={styles.brandLogo} />
          <img src={ibmLogo} alt="IBM" className={styles.brandLogo} />
          <img src={icmLogo} alt="ICM Paris" className={styles.brandLogo} />
        </div>
        <Heading as="h1" className={styles.heroTitle}>
          {siteConfig.title}
        </Heading>
        <p className={styles.heroSubtitle}>
          Prediction et debiaisage du score moteur MDS-UPDRS (true OFF) dans la maladie de Parkinson.
          Projet d apprentissage automatique encadre par Probabl et IBM pour l ESILV.
        </p>
        <div className={styles.heroButtons}>
          <Link
            className={styles.primaryBtn}
            to="/doc/introduction">
            Consulter la Documentation
          </Link>
          <a
            className={styles.secondaryBtn}
            href="#rapports">
            Rapports Exploratoires (EDA)
          </a>
          <a
            className={styles.secondaryBtn}
            href="https://www.kaggle.com/competitions/bobathon-esilv/"
            target="_blank"
            rel="noopener noreferrer">
            Competition Kaggle
          </a>
        </div>
      </div>
    </header>
  );
}

function StatsSection() {
  const stats = [
    {
      value: '44 590',
      label: 'Visites d Entrainement',
      detail: '5 576 patients distincts (X_train)',
    },
    {
      value: '11 013',
      label: 'Visites de Test',
      detail: '1 395 patients holdout (X_test)',
    },
    {
      value: '0 %',
      label: 'Perte Totale',
      detail: '100% des visites ont au moins un score moteur',
    },
    {
      value: '5 Plis',
      label: 'GroupKFold Patient',
      detail: 'Partitionnement strict par patient_id',
    },
  ];

  return (
    <div className="container">
      <div className={styles.statsRow}>
        {stats.map((s, idx) => (
          <div key={idx} className={styles.statItem}>
            <div className={styles.statValue}>{s.value}</div>
            <div className={styles.statLabel}>{s.label}</div>
            <div className={styles.statDetail}>{s.detail}</div>
          </div>
        ))}
      </div>
    </div>
  );
}

function ReportsSection() {
  const reportTrainUrl = useBaseUrl('/reports/eda_visits_train.html');
  const reportTestUrl = useBaseUrl('/reports/eda_X_test.html');

  return (
    <section id="rapports" className={styles.section}>
      <div className="container">
        <div className={styles.sectionHeader}>
          <div className={styles.sectionTag}>Donnees & Exploration</div>
          <Heading as="h2" className={styles.sectionTitle}>
            Rapports d Analyse Exploratoire
          </Heading>
          <p className={styles.sectionDesc}>
            Profilages interactifs generes lors du Day 1. Analyse multivariee,
            distributions des scores moteurs et mecanismes non-MCAR des valeurs manquantes.
          </p>
        </div>

        <div className={styles.reportsGrid}>
          <div className={styles.reportItem}>
            <div>
              <Heading as="h3" className={styles.reportItemTitle}>
                Visites d Entrainement (Train Set)
              </Heading>
              <p className={styles.reportItemDesc}>
                44 590 consultations et 5 576 patients. Distribution symetrique de la cible
                true OFF (moyenne 37.47, ecart-type 16.50), correlations Pearson (off a +0.87)
                et variations des posologies LEDD par cohorte clinique.
              </p>
            </div>
            <a href={reportTrainUrl} target="_blank" rel="noopener noreferrer" className={styles.reportLink}>
              Ouvrir le rapport interactif (eda_visits_train.html) &rarr;
            </a>
          </div>

          <div className={styles.reportItem}>
            <div>
              <Heading as="h3" className={styles.reportItemTitle}>
                Visites de Test (Test Set)
              </Heading>
              <p className={styles.reportItemDesc}>
                11 013 consultations et 1 395 patients non vus. Verification de la coherence
                structurelle, de l absence de chevauchement de patients et des distributions
                avant soumission finale.
              </p>
            </div>
            <a href={reportTestUrl} target="_blank" rel="noopener noreferrer" className={styles.reportLink}>
              Ouvrir le rapport interactif (eda_X_test.html) &rarr;
            </a>
          </div>
        </div>
      </div>
    </section>
  );
}

function BenchmarkSection() {
  const models = [
    {
      key: '01_dummy',
      name: 'Dummy Mean Regressor',
      description: 'Moyenne empirique globale (plancher d evaluation de reference)',
      rmse: '~16.50',
      status: 'Socle minimal',
    },
    {
      key: '02_ridge',
      name: 'Ridge Regressor (alpha=10)',
      description: 'Regression lineaire regularisee sur variables continues',
      rmse: '-30 %',
      status: 'Valide',
    },
    {
      key: '03_hgbr',
      name: 'HistGradientBoosting + skrub',
      description: 'Arbres avec gestion native des NaN et tabular_pipeline',
      rmse: '-50 %',
      status: 'Valide',
    },
    {
      key: '04_features',
      name: 'Features Temporelles Intra-Patient',
      description: 'Dynamique temporelle de suivi et retards medicamenteux',
      rmse: '-60 %',
      status: 'Valide',
    },
    {
      key: '05_rater',
      name: 'Correction du Biais Evaluateur',
      description: 'Integration du biais systematique du clinicien (rater_id)',
      rmse: '-62 %',
      status: 'Valide',
    },
    {
      key: '07_ensemble',
      name: 'Stacking Ensemble (Ridge + HGBR)',
      description: 'Meta-modele Ridge entraine sur splits groupes PatientGroupKFold',
      rmse: '-67 %',
      status: 'Performant',
    },
  ];

  return (
    <section className={styles.sectionAlt}>
      <div className="container">
        <div className={styles.sectionHeader}>
          <div className={styles.sectionTag}>Modelisation ML</div>
          <Heading as="h2" className={styles.sectionTitle}>
            Progression des Modeles et Benchmarks
          </Heading>
          <p className={styles.sectionDesc}>
            Hierarchie experimentale des modeles developpes dans src/, traces avec skore
            et publies sur Skore Hub.
          </p>
        </div>

        <div className={styles.tableContainer}>
          <table className={styles.benchTable}>
            <thead>
              <tr>
                <th>Cle Hub</th>
                <th>Modele</th>
                <th>Approche Algorithmique</th>
                <th>Impact RMSE</th>
                <th>Statut</th>
              </tr>
            </thead>
            <tbody>
              {models.map((m, idx) => (
                <tr key={idx}>
                  <td><span className={styles.codeKey}>{m.key}</span></td>
                  <td><strong>{m.name}</strong></td>
                  <td>{m.description}</td>
                  <td>{m.rmse}</td>
                  <td><span className={styles.pillNeutral}>{m.status}</span></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </section>
  );
}

function FiguresSection() {
  const figures = [
    {
      img: useBaseUrl('/figures/figure1_anatomy.png'),
      title: 'Anatomie & Voies Dopaminergiques',
      desc: 'Noyaux gris centraux et mecanismes synaptiques de la maladie.',
    },
    {
      img: useBaseUrl('/figures/figure2_treatment_response.png'),
      title: 'Fenetre Therapeutique',
      desc: 'Evolution de la reponse a la Levodopa et epuisement wearing-off.',
    },
    {
      img: useBaseUrl('/figures/figure3_mds_updrs.png'),
      title: 'Examen MDS-UPDRS',
      desc: 'Standardisation de l examen moteur et cotation de gravite (0 a 132).',
    },
    {
      img: useBaseUrl('/figures/figure4_score_evolution.png'),
      title: 'Trajectoire Patient & true OFF',
      desc: 'Correction des biais d observation pour reconstituer la cible impartiale.',
    },
  ];

  return (
    <section className={styles.section}>
      <div className="container">
        <div className={styles.sectionHeader}>
          <div className={styles.sectionTag}>Sciences & Contexte</div>
          <Heading as="h2" className={styles.sectionTitle}>
            Comprendre le Defi Neurologique
          </Heading>
          <p className={styles.sectionDesc}>
            Synthese visuelle des mecanismes physiopathologiques guidant le feature engineering.
          </p>
        </div>

        <div className={styles.figuresGrid}>
          {figures.map((f, idx) => (
            <div key={idx} className={styles.figureBlock}>
              <img src={f.img} alt={f.title} className={styles.figureImg} />
              <Heading as="h3" className={styles.figureTitle}>{f.title}</Heading>
              <p className={styles.figureCaption}>{f.desc}</p>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

function TeamSection() {
  const members = [
    { name: 'Baptiste', role: 'Feature Engineering & Data Science' },
    { name: 'Maxime', role: 'Machine Learning & Model Ensembling' },
    { name: 'Louis', role: 'Architecture, Evaluation & Documentation' },
  ];

  return (
    <section className={styles.teamSection}>
      <div className="container">
        <div style={{ textAlign: 'center' }}>
          <div className={styles.sectionTag}>Equipe Projet</div>
          <Heading as="h2" className={styles.sectionTitle}>
            Etudiants ESILV - Majeure Data & IA
          </Heading>
          <p className={styles.sectionDesc} style={{ margin: '0 auto' }}>
            Bobathon ESILV - Hackathon Machine Learning IBM x Probabl
          </p>
        </div>

        <div className={styles.teamList}>
          {members.map((m, idx) => (
            <div key={idx} className={styles.teamMember}>
              <div className={styles.teamMemberName}>{m.name}</div>
              <div className={styles.teamMemberRole}>{m.role}</div>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}

export default function Home() {
  const {siteConfig} = useDocusaurusContext();
  return (
    <Layout
      title="Accueil"
      description="Plateforme de presentation et documentation du Bobathon ESILV - Challenge Machine Learning Parkinson">
      <HomepageHero />
      <main>
        <StatsSection />
        <ReportsSection />
        <BenchmarkSection />
        <FiguresSection />
        <TeamSection />
      </main>
    </Layout>
  );
}
