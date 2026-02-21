"""Generate a sample ALS corpus for tokenizer training development.

Creates a representative synthetic corpus containing the same types of content
that the Phase 1 scrapers would collect and Phase 2 pipeline would process.
This is used when raw scraped data is unavailable (e.g., fresh environment
without API access). The corpus includes PubMed-style abstracts, clinical trial
descriptions, educational content, Wikipedia-style articles, and FDA drug
information, all focused on ALS/MND topics.

The generated text is separated by <|endoftext|> markers matching the Phase 2
output convention, and is written to data/processed/train.txt and val.txt.
"""

import argparse
import random
import sys
from pathlib import Path


# Representative ALS corpus content organized by source type
PUBMED_ABSTRACTS = [
    "Amyotrophic lateral sclerosis (ALS) is a progressive neurodegenerative disease affecting motor neurons in the brain and spinal cord. The hallmark of ALS is the degeneration of both upper motor neurons (UMN) and lower motor neurons (LMN), leading to muscle weakness, atrophy, fasciculations, and spasticity. The median survival from symptom onset is 2 to 5 years, with respiratory failure being the primary cause of death. Approximately 90-95% of cases are sporadic (sALS), while the remaining 5-10% are familial (fALS), with over 30 genes implicated in disease pathogenesis. The SOD1 gene was the first identified ALS gene in 1993, encoding copper-zinc superoxide dismutase. Mutations in SOD1 account for approximately 20% of familial ALS cases and 1-2% of sporadic cases. The most common genetic cause of ALS is a hexanucleotide repeat expansion in the C9orf72 gene, accounting for 40% of familial and 5-10% of sporadic ALS in European populations.",

    "Riluzole remains the first FDA-approved drug for ALS, demonstrating a modest extension of survival by 2-3 months in clinical trials. The mechanism of action involves inhibition of glutamate release, sodium channel blockade, and interference with intracellular events following transmitter binding at excitatory amino acid receptors. Edaravone (Radicava) was approved in 2017 as a free radical scavenger, though its efficacy has been debated. More recently, tofersen, an antisense oligonucleotide targeting SOD1 mRNA, was approved under accelerated approval for SOD1-ALS. The AMX0035 combination of sodium phenylbutyrate and taurursodiol (Relyvrio) received approval in 2022, though the confirmatory trial did not meet its primary endpoint.",

    "The TDP-43 protein, encoded by the TARDBP gene, is a major component of ubiquitinated cytoplasmic inclusions found in approximately 97% of ALS cases. TDP-43 pathology represents a unifying feature of the ALS-frontotemporal dementia (FTD) spectrum. Under normal conditions, TDP-43 is predominantly nuclear and involved in RNA processing, including splicing, transport, and stability. In ALS, TDP-43 mislocalizes from the nucleus to the cytoplasm, where it forms phosphorylated and ubiquitinated aggregates. This loss of nuclear function and gain of cytoplasmic toxic function contribute to motor neuron degeneration. Recent studies have identified cryptic exon inclusion as a key pathological mechanism downstream of TDP-43 dysfunction.",

    "Electromyography (EMG) plays a crucial role in the diagnosis of ALS, demonstrating evidence of ongoing denervation and reinnervation in clinically affected and unaffected regions. The revised El Escorial criteria and the Awaji criteria provide diagnostic frameworks for ALS, requiring evidence of UMN and LMN degeneration in multiple body regions. The Awaji criteria incorporate fasciculation potentials as evidence of denervation, improving diagnostic sensitivity. Nerve conduction studies are performed to exclude mimicking conditions such as multifocal motor neuropathy with conduction block. The ALSFRS-R (ALS Functional Rating Scale-Revised) is the most widely used clinical outcome measure, assessing 12 functional activities rated on a 5-point scale.",

    "Frontotemporal dementia (FTD) occurs in approximately 15% of ALS patients, with an additional 35% showing cognitive or behavioral impairment that does not meet full FTD criteria. The C9orf72 hexanucleotide repeat expansion is the most common genetic link between ALS and FTD, suggesting these diseases represent a clinical spectrum. Patients with ALS-FTD typically have a worse prognosis than those with ALS alone. Behavioral variant FTD (bvFTD) is the most common FTD phenotype in ALS patients, characterized by personality changes, disinhibition, apathy, and executive dysfunction. The overlap between ALS and FTD has important implications for clinical management, genetic counseling, and therapeutic development.",

    "Respiratory function monitoring is essential in ALS management. Forced vital capacity (FVC) and sniff nasal inspiratory pressure (SNIP) are the most commonly used measures of respiratory function. Non-invasive ventilation (NIV) improves quality of life and extends survival when initiated at FVC below 80% or when symptomatic respiratory insufficiency develops. Diaphragmatic pacing has been investigated but a randomized trial showed decreased survival compared to standard care. Dysphagia management involves dietary modification, and percutaneous endoscopic gastrostomy (PEG) is recommended when weight loss exceeds 10% or swallowing becomes unsafe. Sialorrhea can be managed with anticholinergic medications or botulinum toxin injections.",

    "The FUS (Fused in Sarcoma) gene encodes an RNA-binding protein that, like TDP-43, is normally nuclear and involved in RNA metabolism. FUS mutations cause approximately 4% of familial ALS and are associated with a younger age of onset. Unlike TDP-43, FUS mutations typically lead to FUS-immunoreactive inclusions rather than TDP-43 inclusions. The TBK1 (TANK-binding kinase 1) gene is involved in autophagy and neuroinflammation, with loss-of-function mutations identified in both ALS and FTD. OPTN (Optineurin) mutations are rare causes of familial ALS, acting through disruption of autophagy and NF-kB signaling pathways. NEK1 is a risk gene for sporadic ALS, with variants identified in approximately 3% of cases.",

    "Stem cell therapy for ALS remains largely experimental. Mesenchymal stem cells (MSCs), neural progenitor cells, and induced pluripotent stem cell (iPSC)-derived motor neurons have been investigated in clinical trials. The NurOwn technology uses autologous bone marrow-derived MSCs engineered to secrete neurotrophic factors. While Phase 2 results showed some biomarker improvements, the Phase 3 trial did not meet its primary endpoint. iPSC technology has revolutionized ALS modeling, enabling the generation of patient-specific motor neurons that recapitulate disease phenotypes in vitro. These cellular models facilitate drug screening and mechanistic studies without the confounds of postmortem tissue analysis.",

    "Biomarkers for ALS diagnosis and prognosis are an active area of research. Neurofilament light chain (NfL) is the most promising fluid biomarker, with elevated levels in cerebrospinal fluid (CSF) and blood correlating with disease progression and survival. Phosphorylated neurofilament heavy chain (pNfH) also shows diagnostic utility. Urinary p75ECD (p75 neurotrophin receptor extracellular domain) is another emerging biomarker. MRI-based biomarkers include diffusion tensor imaging (DTI) showing reduced fractional anisotropy in the corticospinal tract. Emerging techniques such as positron emission tomography (PET) with microglia-specific tracers can detect neuroinflammation in vivo.",

    "Motor neuron disease (MND) encompasses a spectrum of conditions including ALS, primary lateral sclerosis (PLS), progressive muscular atrophy (PMA), and progressive bulbar palsy (PBP). PLS is characterized by isolated UMN degeneration, while PMA shows isolated LMN involvement. These conditions may represent phenotypic variants of the same underlying disease process, with some PLS and PMA patients eventually developing features of ALS. Kennedy disease (spinal and bulbar muscular atrophy, SBMA) is an X-linked condition caused by CAG repeat expansion in the androgen receptor gene that must be distinguished from ALS. Hereditary spastic paraplegia (HSP) can also mimic UMN-predominant ALS.",

    "Glutamate excitotoxicity is one of the earliest proposed mechanisms of motor neuron death in ALS. Excessive glutamate signaling through AMPA and NMDA receptors leads to calcium influx and mitochondrial dysfunction. Motor neurons are particularly vulnerable to excitotoxicity due to their low calcium-buffering capacity and high expression of calcium-permeable AMPA receptors lacking the GluR2 subunit. The excitotoxicity hypothesis provided the rationale for riluzole development. Subsequent research has revealed additional pathogenic mechanisms including oxidative stress, protein aggregation, impaired proteostasis, mitochondrial dysfunction, axonal transport defects, RNA processing abnormalities, and neuroinflammation mediated by reactive astrocytes and microglia.",

    "The ATXN2 gene, encoding ataxin-2, has been identified as a modifier of ALS risk. Intermediate-length polyglutamine expansions (27-33 repeats, compared to normal 22-23) in ATXN2 are associated with increased ALS susceptibility. This discovery emerged from genetic interaction studies with TDP-43 in yeast models. ATXN2 intermediate expansions are found in approximately 5% of ALS patients compared to 1-2% of controls. Longer expansions (>34 repeats) cause spinocerebellar ataxia type 2 (SCA2). The mechanism involves enhanced stress granule formation and TDP-43 mislocalization. This finding has implications for potential therapeutic targeting of the ATXN2-TDP-43 interaction.",

    "Bulbar-onset ALS accounts for approximately 25% of cases and is characterized by initial involvement of muscles controlling speech (dysarthria), swallowing (dysphagia), and tongue movement. Patients with bulbar onset have a worse prognosis than limb-onset ALS, with median survival of 2-3 years compared to 3-5 years for limb onset. Women are more frequently affected by bulbar-onset ALS than men. Pseudobulbar affect (PBA), characterized by involuntary laughing or crying disproportionate to the underlying emotional state, occurs in approximately 50% of ALS patients. The combination of dextromethorphan and quinidine (Nuedexta) is FDA-approved for the treatment of PBA.",

    "Gene therapy approaches for ALS include antisense oligonucleotides (ASOs), RNA interference (RNAi), and adeno-associated virus (AAV)-mediated gene delivery. Tofersen, an ASO targeting SOD1, demonstrated neurofilament reduction in the VALOR trial. ASOs for C9orf72 ALS are in clinical development. AAV-mediated delivery of microRNA against SOD1 has shown efficacy in animal models. CRISPR-Cas9 gene editing has been explored for correcting SOD1 mutations in iPSC-derived motor neurons. The challenge remains achieving adequate distribution to the large motor neuron population throughout the spinal cord and brain. Intrathecal delivery is the current standard route for ASOs.",

    "Multidisciplinary care is considered the gold standard for ALS management, involving neurologists, pulmonologists, speech-language pathologists, physical therapists, occupational therapists, nutritionists, social workers, and palliative care specialists. Studies have demonstrated that multidisciplinary clinic attendance is associated with improved survival and quality of life. Assistive technology includes augmentative and alternative communication (AAC) devices, motorized wheelchairs, and home environmental modifications. Eye-tracking technology enables communication for patients who have lost limb and bulbar function. Advance care planning, including discussions about ventilatory support and end-of-life preferences, should be initiated early in the disease course.",
]

CLINICAL_TRIALS = [
    "A Phase 3, Randomized, Double-Blind, Placebo-Controlled Study to Evaluate the Efficacy and Safety of Tofersen in Adults with Amyotrophic Lateral Sclerosis Associated with SOD1 Mutation. The primary endpoint is change in ALSFRS-R total score from baseline to week 28. Secondary endpoints include change in slow vital capacity (SVC), CSF neurofilament light chain (NfL) concentration, and time to permanent assisted ventilation or death. Eligible participants are adults aged 18 years or older with a confirmed SOD1 mutation and onset of ALS symptoms within 24 months prior to screening. The study excludes participants with FVC less than 50% predicted at screening. Tofersen is administered intrathecally at a dose of 100 mg every 4 weeks following a loading regimen.",

    "A Phase 2/3 Study Evaluating the Safety, Tolerability, and Efficacy of Intravenous Edaravone in Subjects with Amyotrophic Lateral Sclerosis. Treatment cycles consist of daily intravenous infusions over 14 days followed by a 14-day drug-free period. The study uses the ALSFRS-R as the primary outcome measure, with respiratory function, biomarkers, and quality of life as secondary endpoints. Inclusion criteria require definite or probable ALS by revised El Escorial criteria, FVC of 70% or greater, and disease duration of less than 3 years. The study assesses whether edaravone slows functional decline compared to placebo.",

    "Randomized, Controlled Trial of Oral Riluzole in Patients with Amyotrophic Lateral Sclerosis. This multicenter trial evaluates riluzole 100 mg daily versus placebo. The primary endpoint is tracheostomy-free survival. Secondary endpoints include muscle strength (measured by maximum voluntary isometric contraction), FVC, and ALSFRS-R progression rate. The study demonstrates that riluzole extends median survival by approximately 2-3 months, with the effect most pronounced in patients with bulbar-onset disease. Adverse effects include elevated transaminases, asthenia, and gastrointestinal symptoms.",

    "A Phase 1/2 Study of Intrathecally Administered BIIB078, an Antisense Oligonucleotide Targeting C9orf72 mRNA, in Adults with ALS Associated with C9orf72 Repeat Expansion. The study evaluates multiple ascending doses administered intrathecally every 4 or 8 weeks. Primary endpoints are safety, tolerability, and pharmacokinetics. Exploratory biomarker endpoints include poly(GP) dipeptide repeat protein levels in CSF, plasma NfL, and cerebrospinal fluid NfL. Eligible patients must have documented C9orf72 repeat expansion and symptom onset within 36 months.",

    "A Multicenter, Randomized, Double-Blind, Placebo-Controlled Study of AMX0035 (Sodium Phenylbutyrate-Taurursodiol) in Adults with ALS. The CENTAUR trial evaluates the combination of sodium phenylbutyrate 3 g and taurursodiol 1 g administered orally twice daily. The primary endpoint is rate of ALSFRS-R decline over 24 weeks. The study enrolled 137 participants with definite ALS by El Escorial criteria. Results showed a 2.32-point difference in ALSFRS-R decline favoring AMX0035 over placebo. The open-label extension demonstrated a median overall survival benefit of 6.5 months.",
]

EDUCATIONAL_CONTENT = [
    "Amyotrophic lateral sclerosis is a rapidly progressive, fatal neurological disease that attacks the nerve cells responsible for controlling voluntary muscles. The disease belongs to a group of disorders known as motor neuron diseases, which are characterized by the gradual degeneration and death of motor neurons. Motor neurons are nerve cells in the brain, brainstem, and spinal cord that serve as controlling units and communication links between the nervous system and the voluntary muscles of the body. The loss of these neurons leads to progressive muscle weakness, wasting, and paralysis. In ALS, both the upper motor neurons that originate in the brain and the lower motor neurons that connect the brainstem and spinal cord to the muscles gradually degenerate, causing the muscles to weaken, twitch, and waste away. Eventually, the brain loses its ability to initiate and control voluntary movement.",

    "The genetics of ALS have been extensively studied over the past three decades. The discovery of SOD1 mutations in 1993 opened the field of ALS genetics. The SOD1 protein is a ubiquitously expressed enzyme that converts superoxide radicals to hydrogen peroxide and oxygen. Over 180 different mutations in SOD1 have been identified in ALS patients. The A4V mutation is the most common in North America and is associated with a particularly aggressive disease course, with median survival of approximately one year. In contrast, the D90A mutation, common in Scandinavian populations, is associated with a slower progression. The discovery of the C9orf72 repeat expansion in 2011 was a landmark finding, explaining the most common genetic cause of both ALS and FTD. The normal allele contains fewer than 30 hexanucleotide GGGGCC repeats, while pathogenic alleles contain hundreds to thousands of repeats.",

    "Spasticity in ALS results from upper motor neuron degeneration and manifests as velocity-dependent increase in muscle tone, hyperreflexia, clonus, and the Babinski sign. Management includes physical therapy, stretching exercises, and pharmacological agents such as baclofen, tizanidine, and dantrolene. Fasciculations are spontaneous, involuntary muscle twitches caused by lower motor neuron dysfunction. While fasciculations are not unique to ALS and can occur benignly, their presence in combination with other signs of denervation supports the diagnosis. Muscle cramps are common early symptoms of ALS and may precede clinically evident weakness. They result from motor unit instability as denervation progresses.",

    "Nutritional management in ALS is critical because dysphagia, increased metabolic demand, and decreased caloric intake contribute to malnutrition and weight loss. Body mass index decline is an independent predictor of survival. Percutaneous endoscopic gastrostomy placement is recommended when patients can no longer safely meet their nutritional needs orally or when FVC remains above 50% to reduce procedural risk. High-calorie diets, including hypercaloric fatty diets, have shown potential benefits in randomized trials. Respiratory management involves regular monitoring of FVC, maximal inspiratory pressure, and nocturnal oximetry. Non-invasive positive pressure ventilation via bilevel positive airway pressure (BiPAP) is initiated when FVC drops below 50% or when symptoms of nocturnal hypoventilation develop.",
]

WIKIPEDIA_CONTENT = [
    "Amyotrophic lateral sclerosis, also known as motor neuron disease or Lou Gehrig's disease, is a neurodegenerative disease that results in the progressive loss of motor neurons that control voluntary muscles. ALS is the most common form of the motor neuron diseases. Early symptoms include stiff muscles, muscle twitches, and gradual increasing weakness and muscle wasting. Limb-onset ALS begins with weakness in the arms or legs, while bulbar-onset ALS begins with difficulty speaking or swallowing. Most people with ALS die from respiratory failure, usually within two to five years from when the symptoms first appear. About 10% of those affected survive longer than 10 years. The word amyotrophic comes from the Greek language: a means no, myo refers to muscle, and trophic means nourishment. When a muscle has no nourishment, it atrophies or wastes away. Lateral identifies the areas in a person's spinal cord where portions of the nerve cells that signal and control the muscles are located. As this area degenerates, it leads to scarring or hardening (sclerosis) in the region.",

    "The diagnosis of ALS is based primarily on clinical findings, supported by electrodiagnostic studies. The revised El Escorial criteria require the presence of evidence of lower motor neuron degeneration by clinical, electrophysiological, or neuropathological examination, evidence of upper motor neuron degeneration by clinical examination, and progressive spread of symptoms or signs within a region or to other regions. The Awaji-shima electrodiagnostic criteria, proposed in 2008, consider electrophysiological evidence of acute denervation and fasciculation potentials as equivalent to clinical evidence of lower motor neuron signs, improving diagnostic sensitivity. The Gold Coast criteria, published in 2020, provide further refinement, requiring progressive motor impairment documented by history or repeated clinical assessment and evidence of both upper and lower motor neuron dysfunction in at least one body region.",

    "Research into the pathophysiology of ALS has identified several key mechanisms. Protein aggregation and impaired proteostasis involve the accumulation of misfolded proteins, particularly TDP-43, in motor neurons. RNA metabolism dysfunction results from the nuclear depletion of RNA-binding proteins like TDP-43 and FUS. Mitochondrial dysfunction leads to impaired energy production and increased oxidative stress. Glutamate excitotoxicity causes excessive stimulation of motor neurons through overactivation of glutamate receptors. Neuroinflammation involves the activation of astrocytes and microglia, which initially may be neuroprotective but become neurotoxic as the disease progresses. Axonal transport deficits disrupt the movement of organelles and proteins along motor axons, which can extend up to one meter in length. These mechanisms are interconnected and likely act synergistically to cause motor neuron death.",
]

FDA_CONTENT = [
    "Riluzole (Rilutek) was approved by the FDA in 1995 for the treatment of amyotrophic lateral sclerosis. The recommended dose is 50 mg taken every 12 hours. Riluzole is a benzothiazole that inhibits glutamate release, inactivates voltage-dependent sodium channels, and interferes with intracellular events that follow transmitter binding at excitatory amino acid receptors. The drug extends median survival by approximately 2-3 months. Common adverse effects include nausea, asthenia, elevated hepatic transaminases, and dizziness. Hepatotoxicity is the most significant safety concern, requiring liver function monitoring before and during treatment. Riluzole is metabolized primarily by CYP1A2.",

    "Edaravone (Radicava) was approved by the FDA in May 2017 for the treatment of ALS. Edaravone is a free radical scavenger that reduces oxidative stress, a proposed mechanism of motor neuron death in ALS. The drug is administered as an intravenous infusion over 60 minutes. Treatment cycles consist of initial daily dosing for 14 days followed by a 14-day drug-free period, then on 10 of 14 days followed by a 14-day drug-free period. An oral formulation (Radicava ORS) was subsequently approved. Clinical trial data showed that edaravone reduced the decline in ALSFRS-R scores by 33% compared to placebo in a selected population of early ALS patients.",

    "The FDA granted accelerated approval to tofersen (Qalsody) in April 2023 for the treatment of ALS in adults who have a mutation in the superoxide dismutase 1 (SOD1) gene. Tofersen is an antisense oligonucleotide designed to reduce the production of SOD1 protein by binding to SOD1 mRNA and promoting its degradation. The drug is administered intrathecally by lumbar puncture. The recommended dosage is 100 mg administered via intrathecal injection, given as three loading doses at 14-day intervals followed by maintenance doses every 28 days. The most common adverse reactions include pain, fatigue, arthralgia, and fall. Serious adverse events include myelitis and papilledema.",
]

PATIENT_NARRATIVE_STYLE = [
    "Living with ALS has fundamentally changed my understanding of what it means to adapt. Six months after my diagnosis, the fasciculations in my hands had progressed to noticeable weakness, making simple tasks like buttoning a shirt or opening a jar increasingly difficult. My neurologist started me on riluzole, and while I experienced some nausea initially, it settled after the first few weeks. The multidisciplinary clinic has been invaluable, connecting me with a speech therapist who helped me bank my voice before dysarthria makes my speech unintelligible. Physical therapy three times per week helps maintain range of motion and reduces the painful spasticity in my legs.",

    "When my father was diagnosed with familial ALS linked to a SOD1 mutation, our family faced the difficult decision of whether to pursue genetic testing. Knowing that we each had a 50% chance of carrying the same mutation weighed heavily on all of us. Genetic counseling helped us understand that carrying the mutation does not guarantee developing ALS, as penetrance is incomplete for many SOD1 variants. Two of my siblings chose to be tested, while I decided to wait. The availability of tofersen for SOD1-ALS has changed the calculus for many families like ours, offering the first mutation-specific treatment option.",
]


def generate_corpus(output_dir: str, seed: int = 42, multiplier: int = 3) -> dict:
    """Generate a synthetic ALS corpus for tokenizer training.

    Args:
        output_dir: Directory to write train.txt and val.txt.
        seed: Random seed for reproducibility.
        multiplier: How many times to replicate and paraphrase content
            to reach a reasonable corpus size.

    Returns:
        Dictionary with corpus statistics.
    """
    random.seed(seed)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Collect all documents
    all_docs = []

    # Add all source types
    for doc in PUBMED_ABSTRACTS:
        all_docs.append(doc)
    for doc in CLINICAL_TRIALS:
        all_docs.append(doc)
    for doc in EDUCATIONAL_CONTENT:
        all_docs.append(doc)
    for doc in WIKIPEDIA_CONTENT:
        all_docs.append(doc)
    for doc in FDA_CONTENT:
        all_docs.append(doc)
    for doc in PATIENT_NARRATIVE_STYLE:
        all_docs.append(doc)

    # Create expanded corpus by replicating with minor variations
    expanded = []
    for i in range(multiplier):
        for doc in all_docs:
            if i == 0:
                expanded.append(doc)
            else:
                # Add slight variations (sentence reordering within paragraphs)
                sentences = doc.split(". ")
                if len(sentences) > 3:
                    # Shuffle middle sentences, keep first and last
                    middle = sentences[1:-1]
                    random.shuffle(middle)
                    varied = [sentences[0]] + middle + [sentences[-1]]
                    expanded.append(". ".join(varied))
                else:
                    expanded.append(doc)

    # Shuffle and split 90/10
    random.shuffle(expanded)
    split_idx = int(len(expanded) * 0.9)
    train_docs = expanded[:split_idx]
    val_docs = expanded[split_idx:]

    # Write files with <|endoftext|> separators
    separator = "<|endoftext|>"

    train_text = separator.join(train_docs)
    val_text = separator.join(val_docs)

    train_path = out / "train.txt"
    val_path = out / "val.txt"

    train_path.write_text(train_text, encoding="utf-8")
    val_path.write_text(val_text, encoding="utf-8")

    stats = {
        "total_documents": len(expanded),
        "train_documents": len(train_docs),
        "val_documents": len(val_docs),
        "train_chars": len(train_text),
        "val_chars": len(val_text),
        "train_file": str(train_path),
        "val_file": str(val_path),
    }
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Generate a sample ALS corpus for tokenizer training"
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed",
        help="Output directory (default: data/processed)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    parser.add_argument(
        "--multiplier",
        type=int,
        default=3,
        help="Content multiplier for corpus size (default: 3)",
    )
    args = parser.parse_args()

    stats = generate_corpus(args.output_dir, args.seed, args.multiplier)

    print(f"Corpus generated:")
    print(f"  Total documents: {stats['total_documents']}")
    print(f"  Train: {stats['train_documents']} docs, {stats['train_chars']:,} chars")
    print(f"  Val: {stats['val_documents']} docs, {stats['val_chars']:,} chars")
    print(f"  Train file: {stats['train_file']}")
    print(f"  Val file: {stats['val_file']}")


if __name__ == "__main__":
    main()
