import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.datasets import cifar10
import pickle
import os
import pandas as pd

st.set_page_config(
    page_title="CIFAR-10 — Projet ML",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

CLASSES = ['avion', 'automobile', 'oiseau', 'chat', 'cerf',
           'chien', 'grenouille', 'cheval', 'bateau', 'camion']
EMOJIS = ['✈️', '🚗', '🐦', '🐱', '🦌', '🐶', '🐸', '🐴', '⛵', '🚛']
MODELS_DIR = "models"
OUTPUTS_DIR = "outputs"

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;700;800&family=JetBrains+Mono:wght@400;600&display=swap');
html, body, [class*="css"] { font-family: 'Syne', sans-serif !important; }
.main-title { font-size: 2.2rem; font-weight: 800; letter-spacing: -0.03em; margin-bottom: 0.2rem; }
.subtitle { font-size: 0.9rem; opacity: 0.5; margin-bottom: 1.5rem; }
</style>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner="Chargement de CIFAR-10…")
def load_data():
    (X_train, y_train), (X_test, y_test) = cifar10.load_data()
    X_train = X_train.astype("float32") / 255.0
    X_test  = X_test.astype("float32")  / 255.0
    y_train = y_train.flatten()
    y_test  = y_test.flatten()
    return X_train, y_train, X_test, y_test

@st.cache_resource(show_spinner="Chargement du modèle…")
def load_model_cached(path):
    return keras.models.load_model(path)

@st.cache_data(show_spinner="Chargement des historiques…")
def load_history():
    acc, loss = {}, {}
    for name, fname in [("acc","data_acc.pickle"),("loss","data_loss.pickle")]:
        path = os.path.join(OUTPUTS_DIR, fname)
        if os.path.exists(path):
            with open(path,"rb") as f:
                if name == "acc": acc = pickle.load(f)
                else: loss = pickle.load(f)
    return acc, loss

with st.sidebar:
    st.markdown("### 🧠 CIFAR-10")
    st.markdown("---")
    page = st.radio("Navigation", [
        "🏠 Introduction",
        "🗂️ Données",
        "📊 Baseline MLP",
        "🧠 Modèles CNN",
        "⚖️ Comparaison",
        "❌ Analyse des erreurs",
        "📖 Glossaire",
    ], label_visibility="collapsed")
    st.markdown("---")
    st.caption("BUT SD — Apprentissage pour l'IA\nMars 2026")

# ── INTRO ──────────────────────────────────────────────────────────
if page == "🏠 Introduction":
    st.markdown('<div class="main-title">Classification CIFAR-10</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Projet BUT SD — Apprentissage pour l\'IA — Mars 2026</div>', unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Images totales","60 000","50k train + 10k test")
    c2.metric("Classes","10","avion → camion")
    c3.metric("Taille image","32×32×3","pixels RGB")
    c4.metric("Modèles testés","5","1 MLP + 4 CNN")

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("#### 🎯 Objectif")
        st.markdown("""Entraîner des réseaux de neurones capables de **classer automatiquement** des images en 10 catégories.

On compare progressivement :
- Un **MLP basique** (référence)
- Quatre **CNN** avec filtres 3×3 vs 5×5, avec/sans data augmentation""")
    with col2:
        st.markdown("#### ✅ Étapes réalisées")
        for step in ["Exploration & visualisation CIFAR-10","Normalisation des données","Baseline MLP entraîné & analysé","4 CNN comparés","Analyse des erreurs & classes difficiles"]:
            st.markdown(f"✅ {step}")

    st.markdown("---")
    st.markdown("#### 🏆 Résultats obtenus")
    col1,col2,col3 = st.columns(3)
    col1.metric("Baseline MLP","42.4%","accuracy test")
    col2.metric("Meilleur CNN","~74%","+31.6 pts vs baseline")
    col3.info("💡 Les CNN sont bien supérieurs au MLP pour les images car ils préservent la **structure spatiale** des pixels.")

# ── DONNÉES ────────────────────────────────────────────────────────
elif page == "🗂️ Données":
    st.markdown('<div class="main-title">Exploration des données</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Comprendre CIFAR-10 avant de modéliser</div>', unsafe_allow_html=True)

    X_train, y_train, X_test, y_test = load_data()
    tab1, tab2, tab3 = st.tabs(["📐 Structure", "🖼️ Images", "📊 Répartition"])

    with tab1:
        col1,col2 = st.columns(2)
        with col1:
            st.markdown("#### Forme des données")
            st.code(f"X_train : {X_train.shape}\ny_train : {y_train.shape}\nX_test  : {X_test.shape}\ny_test  : {y_test.shape}")
            st.markdown("#### Normalisation")
            st.code("X = X.astype('float32') / 255.0\n# pixels : 0–255 → 0.0–1.0")
            st.success(f"Min : {X_train.min():.1f}  |  Max : {X_train.max():.1f}  ✅")
        with col2:
            st.markdown("#### Les 10 classes")
            for i,(cls,em) in enumerate(zip(CLASSES,EMOJIS)):
                st.markdown(f"`{i}` {em} **{cls}** — 5 000 images train")

    with tab2:
        st.markdown("#### 10 exemples par classe")
        fig, axes = plt.subplots(10,10,figsize=(14,14))
        for classe in range(10):
            indices = np.where(y_train==classe)[0][:10]
            for j in range(10):
                axes[classe][j].imshow(X_train[indices[j]])
                axes[classe][j].axis('off')
                if j==0:
                    axes[classe][0].set_ylabel(f"{EMOJIS[classe]} {CLASSES[classe]}",
                                               rotation=0,labelpad=70,fontsize=9,va='center')
        plt.tight_layout()
        st.pyplot(fig); plt.close()

    with tab3:
        st.markdown("#### Répartition des classes (parfaitement équilibrée)")
        counts = [int(np.sum(y_train==i)) for i in range(10)]
        fig,ax = plt.subplots(figsize=(10,4))
        ax.bar([f"{EMOJIS[i]}\n{CLASSES[i]}" for i in range(10)],counts,color='#185FA5',alpha=0.8,edgecolor='white')
        ax.axhline(5000,color='red',linestyle='--',alpha=0.5,label='5000/classe')
        for i,c in enumerate(counts): ax.text(i,c+50,str(c),ha='center',fontsize=9)
        ax.legend(); ax.set_title("Nombre d'images par classe (train)")
        st.pyplot(fig); plt.close()
        st.info("Dataset parfaitement équilibré → pas de biais lié au déséquilibre des classes.")

# ── BASELINE ───────────────────────────────────────────────────────
elif page == "📊 Baseline MLP":
    st.markdown('<div class="main-title">Modèle Baseline — MLP</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Perceptron Multi-Couches — référence de comparaison</div>', unsafe_allow_html=True)

    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Accuracy train","47.9%")
    c2.metric("Accuracy test","42.4%",delta="-5.5%",delta_color="inverse")
    c3.metric("Paramètres","98 666")
    c4.metric("Overfitting","Léger ⚠️")

    st.markdown("---")
    col1,col2 = st.columns(2)
    with col1:
        st.markdown("#### Architecture")
        st.code("""model = keras.Sequential([
    Input(shape=(32, 32, 3)),
    Flatten(),              # 32×32×3 → 3072
    Dense(32, activation='relu'),
    Dense(10, activation='softmax')
])
model.compile(
    optimizer='sgd',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)""", language="python")
    with col2:
        st.markdown("#### Résumé des couches")
        st.dataframe(pd.DataFrame({
            "Couche":["Flatten","Dense(32)","Dense(10)"],
            "Sortie":["(None, 3072)","(None, 32)","(None, 10)"],
            "Paramètres":["0","98 336","330"]
        }), hide_index=True, use_container_width=True)
        st.markdown("**Calcul Dense(32) :** `3072 × 32 + 32 = 98 336`")

    st.markdown("---")
    st.markdown("#### Courbes d'apprentissage (depuis pickle)")
    acc, loss_h = load_history()

    baseline_keys_acc  = [k for k in acc  if "baseline" in k.lower()]
    baseline_keys_loss = [k for k in loss_h if "baseline" in k.lower()]

    if baseline_keys_acc:
        col1,col2 = st.columns(2)
        with col1:
            st.markdown("Accuracy"); st.line_chart(pd.DataFrame({k:acc[k] for k in baseline_keys_acc}))
        with col2:
            st.markdown("Loss"); st.line_chart(pd.DataFrame({k:loss_h[k] for k in baseline_keys_loss}))
    else:
        st.info("Les clés pickle du baseline ne contiennent pas 'baseline' — affichage de toutes les courbes dans la page Comparaison.")

    st.warning("**📊 Analyse :** Le MLP atteint ~48% train mais seulement ~42% test → **overfitting léger**. La courbe test est instable car le Flatten perd la **structure spatiale** des images.")

# ── CNN ────────────────────────────────────────────────────────────
elif page == "🧠 Modèles CNN":
    st.markdown('<div class="main-title">Modèles CNN</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">4 architectures convolutives comparées</div>', unsafe_allow_html=True)

    CNNS = [
        ("CNN1 — 3×3", "(3,3)", False, "CNN1_3x3.keras"),
        ("CNN2 — 5×5", "(5,5)", False, "CNN2_5x5.keras"),
        ("CNN3 — 3×3 + Aug.", "(3,3)", True, "CNN3_3x3_augmented.keras"),
        ("CNN4 — 5×5 + Aug.", "(5,5)", True, "CNN4_5x5_augmented.keras"),
    ]
    tabs = st.tabs([c[0] for c in CNNS])

    for tab, (name, kernel, aug, fname) in zip(tabs, CNNS):
        with tab:
            col1,col2 = st.columns(2)
            with col1:
                aug_code = "\n    RandomFlip('horizontal'),\n    RandomTranslation(0.1, 0.1),\n    RandomZoom(0.1)," if aug else ""
                st.code(f"""model = keras.Sequential([
    Input(shape=(32, 32, 3)),{aug_code}
    Conv2D(32, {kernel}, padding='same', activation='relu'),
    MaxPooling2D(2),
    Conv2D(64, {kernel}, activation='relu'),
    MaxPooling2D(2),
    Flatten(),
    Dropout(0.5),
    Dense(10, activation='softmax')
])
model.compile(optimizer=Adam(lr=0.001), ...)""", language="python")
            with col2:
                st.markdown("**Pourquoi ces choix ?**")
                st.markdown(f"""
- **Conv2D {kernel}** → filtre glisse sur l'image, détecte motifs locaux
- **MaxPooling** → divise résolution par 2, garde l'essentiel
- **Dropout(0.5)** → réduit l'overfitting
- **Softmax** → 10 probabilités en sortie
- **Adam lr=0.001** → optimizer adaptatif
""")
                if aug: st.info("🔄 **Data Augmentation** : flip, zoom, décalage → meilleure généralisation sur beaucoup d'epochs.")
                mpath = os.path.join(MODELS_DIR, fname)
                if os.path.exists(mpath):
                    m = load_model_cached(mpath)
                    st.success(f"✅ Modèle chargé : `{fname}`")
                    with st.expander("Voir le summary"):
                        lines = []
                        m.summary(print_fn=lambda x: lines.append(x))
                        st.code("\n".join(lines))
                else:
                    st.warning(f"⚠️ `{fname}` non trouvé dans `{MODELS_DIR}/`")

# ── COMPARAISON ────────────────────────────────────────────────────
elif page == "⚖️ Comparaison":
    st.markdown('<div class="main-title">Comparaison des modèles</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Accuracy & Loss — tous modèles</div>', unsafe_allow_html=True)

    acc, loss_h = load_history()

    if acc and loss_h:
        tab1, tab2 = st.tabs(["📈 Accuracy", "📉 Loss"])
        with tab1:
            col1,col2 = st.columns(2)
            train_acc = {k:v for k,v in acc.items() if "train" in k}
            val_acc   = {k:v for k,v in acc.items() if "val"   in k}
            with col1: st.markdown("**Train**"); st.line_chart(pd.DataFrame(train_acc))
            with col2: st.markdown("**Test**");  st.line_chart(pd.DataFrame(val_acc))
        with tab2:
            col1,col2 = st.columns(2)
            train_l = {k:v for k,v in loss_h.items() if "train" in k}
            val_l   = {k:v for k,v in loss_h.items() if "val"   in k}
            with col1: st.markdown("**Train**"); st.line_chart(pd.DataFrame(train_l))
            with col2: st.markdown("**Test**");  st.line_chart(pd.DataFrame(val_l))
    else:
        st.warning("⚠️ Fichiers pickle non trouvés dans `outputs/`. Lance d'abord le notebook.")

    st.markdown("---")
    st.markdown("#### Tableau récapitulatif")
    st.dataframe(pd.DataFrame({
        "Modèle":       ["Baseline MLP","CNN1 3×3","CNN2 5×5","CNN3 3×3+Aug.","CNN4 5×5+Aug."],
        "Optimizer":    ["SGD","Adam","Adam","Adam","Adam"],
        "Augmentation": ["❌","❌","❌","✅","✅"],
        "Acc. Train":   ["47.9%","~78%","~76%","~75%","~74%"],
        "Acc. Test":    ["42.4%","~71%","~74%","~70%","~72%"],
        "Overfitting":  ["Léger","Modéré","Modéré","Faible","Faible"],
    }), hide_index=True, use_container_width=True)

    st.markdown("---")
    col1,col2 = st.columns(2)
    with col1:
        st.success("✅ **CNN >> MLP** : +30 pts d'accuracy. Les CNN préservent la structure spatiale.")
        st.info("🔍 **3×3 vs 5×5** : filtres 5×5 légèrement meilleurs ici.")
    with col2:
        st.warning("⚠️ **Data Augmentation** : moins efficace sur 20 epochs. Images 32×32 trop petites.")
        st.info("📈 Avec 50-100 epochs, les modèles augmentés dépasseraient les autres.")

# ── ERREURS ────────────────────────────────────────────────────────
elif page == "❌ Analyse des erreurs":
    st.markdown('<div class="main-title">Analyse des erreurs</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Classes difficiles & exemples mal classés</div>', unsafe_allow_html=True)

    _, _, X_test, y_test = load_data()

    available = {k:v for k,v in {
        "CNN2 5×5":"CNN2_5x5.keras",
        "CNN1 3×3":"CNN1_3x3.keras",
        "CNN3 3×3+Aug.":"CNN3_3x3_augmented.keras",
        "CNN4 5×5+Aug.":"CNN4_5x5_augmented.keras",
    }.items() if os.path.exists(os.path.join(MODELS_DIR,v))}

    if not available:
        st.warning("⚠️ Aucun modèle CNN trouvé dans `models/`.")
    else:
        selected = st.selectbox("Choisir le modèle", list(available.keys()))
        if st.button("🔍 Analyser les erreurs", type="primary"):
            model = load_model_cached(os.path.join(MODELS_DIR, available[selected]))
            with st.spinner("Prédiction…"):
                y_pred = model.predict(X_test, verbose=0).argmax(axis=1)
            errors = [i for i in range(len(y_pred)) if y_pred[i]!=y_test[i]]

            c1,c2,c3 = st.columns(3)
            c1.metric("Erreurs", f"{len(errors)} / {len(y_test)}")
            c2.metric("Accuracy", f"{(1-len(errors)/len(y_test))*100:.2f}%")
            c3.metric("Correctes", f"{len(y_test)-len(errors)}")

            st.markdown("---")
            st.markdown("#### Taux d'erreur par classe")
            err_data = []
            for i,cls in enumerate(CLASSES):
                idx_cls = np.where(y_test==i)[0]
                n_err = sum(1 for j in idx_cls if y_pred[j]!=y_test[j])
                err_data.append({"Classe":f"{EMOJIS[i]} {cls}","Erreurs":n_err,"Total":len(idx_cls),"Taux (%)":round(n_err/len(idx_cls)*100,1)})
            df_err = pd.DataFrame(err_data).sort_values("Taux (%)",ascending=False)

            fig,ax = plt.subplots(figsize=(10,4))
            colors = ['#E24B4A' if x>35 else '#BA7517' if x>25 else '#3B6D11' for x in df_err["Taux (%)"]]
            ax.barh(df_err["Classe"],df_err["Taux (%)"],color=colors,alpha=0.85)
            ax.axvline(10,color='black',linestyle='--',alpha=0.4,label='Hasard (10%)')
            ax.set_xlabel("Taux d'erreur (%)"); ax.legend()
            plt.tight_layout(); st.pyplot(fig); plt.close()

            st.markdown("---")
            st.markdown("#### Exemples d'images mal classées")
            fig,axes = plt.subplots(3,10,figsize=(15,5))
            for idx_err,ax in zip(errors[:30],axes.flat):
                ax.imshow(X_test[idx_err])
                ax.set_title(f"P:{CLASSES[y_pred[idx_err]][:4]}\nV:{CLASSES[y_test[idx_err]][:4]}",fontsize=7)
                ax.axis('off')
            plt.suptitle("P=Prédit  V=Vrai",fontsize=11)
            plt.tight_layout(); st.pyplot(fig); plt.close()

            st.markdown("---")
            col1,col2 = st.columns(2)
            col1.error("🐱🐶 **Chat & Chien** — les plus confondus\nSilhouette similaire, images 32×32 trop petites pour distinguer les détails.")
            col2.warning("🦌🐴 **Cerf & Cheval** — souvent confondus\nQuadrupèdes avec silhouettes proches à basse résolution.")

# ── GLOSSAIRE ──────────────────────────────────────────────────────
elif page == "📖 Glossaire":
    st.markdown('<div class="main-title">Glossaire</div>', unsafe_allow_html=True)
    st.markdown('<div class="subtitle">Les concepts clés du projet</div>', unsafe_allow_html=True)

    for title, body in [
        ("🧠 Réseau de neurones", "Ensemble de neurones artificiels en couches. Chaque neurone calcule somme pondérée + biais puis applique une activation. Les poids sont ajustés par la backpropagation."),
        ("📐 MLP", "Architecture Dense fully-connected. Pour les images, aplatit les pixels (Flatten) → perd la structure spatiale. Simple mais limité pour les images."),
        ("🖼️ CNN", "Filtres qui glissent sur l'image pour détecter des motifs locaux (bords, textures). Préserve la structure spatiale. Bien supérieur au MLP pour les images."),
        ("⚡ Fonction d'activation", "Non-linéarité appliquée après chaque neurone. Sans elle, empiler des couches = une seule couche linéaire. ReLU = max(0,x). Softmax = probabilités en sortie."),
        ("📉 Fonction de coût (Loss)", "Mesure l'erreur entre prédiction et vrai label. Sparse Categorical Crossentropy pour classification multi-classe. But de l'entraînement = minimiser cette valeur."),
        ("🔄 Backpropagation", "Forward pass → erreur → gradient → w ← w - η·∂L/∂w. Répété des milliers de fois pour ajuster tous les poids."),
        ("🎯 Overfitting", "Modèle trop adapté aux données train, mauvaise généralisation. Visible quand val_accuracy << accuracy. Solutions : Dropout, Data Augmentation."),
        ("🔢 Normalisation", "Pixels [0-255] → [0-1]. Évite les gradients explosifs. X = X / 255.0"),
        ("🎲 Data Augmentation", "Variantes d'images artificielles (flip, zoom, rotation). Réduit overfitting. Moins efficace sur petites images avec peu d'epochs."),
        ("📦 Batch & Epoch", "Batch = groupe d'images traité avant chaque mise à jour (ex: 64). Epoch = passage complet sur toutes les données. 50k/64 = 782 mises à jour/epoch."),
        ("💧 Dropout", "Éteint aléatoirement X% des neurones à chaque itération. Dropout(0.5) = 50% désactivés. Force le réseau à ne pas dépendre d'un seul chemin."),
        ("⚙️ Hyperparamètres vs Paramètres", "Paramètres = poids et biais appris automatiquement. Hyperparamètres = learning rate, epochs, batch_size, architecture → à choisir avant l'entraînement."),
    ]:
        with st.expander(title):
            st.markdown(body)
