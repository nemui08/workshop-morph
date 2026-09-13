const evaluateButton = document.getElementById("btn-evaluate");
const processStatus = document.getElementById("process-status");
const evalBox = document.getElementById("eval-box");
const evalNote = document.getElementById("eval-note");
const evalFormulas = document.getElementById("eval-formulas");
const rocImage = document.getElementById("roc-image");


function n(value) {
    return Number(value).toLocaleString("en-US");
}


function r4(value) {
    return value.toFixed(4);
}


evaluateButton.addEventListener("click", async function () {
    processStatus.textContent = "กำลังวัดผล Workshop...";
    evaluateButton.disabled = true;

    try {
        const response = await fetch("/api/evaluate");
        const result = await response.json();

        if (!result.success) {
            processStatus.textContent = result.message || "วัดผลไม่สำเร็จ";
            return;
        }

        const m = result.with_morph;
        const tp = m.tp;
        const fn = m.fn;
        const fp = m.fp;
        const tn = m.tn;
        const p = tp + fn;
        const neg = fp + tn;
        const pp = tp + fp;
        const pn = fn + tn;
        const total = p + neg;
        const acc = (tp + tn) / total;
        const precision = tp / pp;
        const recall = tp / p;
        const fallout = fp / neg;
        const tnr = tn / neg;
        const fnr = fn / p;
        const f1 = 2 * precision * recall / (precision + recall);

        document.getElementById("cell-p").textContent = n(p);
        document.getElementById("cell-n").textContent = n(neg);
        document.getElementById("cell-tp").textContent = "TP " + n(tp);
        document.getElementById("cell-fn").textContent = "FN " + n(fn);
        document.getElementById("cell-fp").textContent = "FP " + n(fp);
        document.getElementById("cell-tn").textContent = "TN " + n(tn);

        evalNote.textContent =
            "Opening · " + result.total + " ภาพ · เทียบกับยังไม่ทำ morph acc=" +
            result.without_morph.accuracy;

        evalFormulas.innerHTML =
            "<p>Accuracy = (TP + TN) / (P + N) = (" + n(tp) + " + " + n(tn) + ") / (" + n(p) + " + " + n(neg) + ") = <b>" + r4(acc) + "</b></p>" +
            "<p>Precision = TP / (TP + FP) = " + n(tp) + " / (" + n(tp) + " + " + n(fp) + ") = <b>" + r4(precision) + "</b></p>" +
            "<p>Recall = TPR = TP / P = " + n(tp) + " / " + n(p) + " = <b>" + r4(recall) + "</b></p>" +
            "<p>Fall-out = FPR = FP / N = " + n(fp) + " / " + n(neg) + " = <b>" + r4(fallout) + "</b></p>" +
            "<p>TNR = TN / N = " + n(tn) + " / " + n(neg) + " = <b>" + r4(tnr) + "</b></p>" +
            "<p>FNR = FN / P = " + n(fn) + " / " + n(p) + " = <b>" + r4(fnr) + "</b></p>" +
            "<p>F1 = 2 · Precision · Recall / (Precision + Recall) = <b>" + r4(f1) + "</b></p>";

        rocImage.src = "data:image/png;base64," + result.roc_image;
        evalBox.style.display = "block";
        processStatus.textContent = "วัดผลสำเร็จ";
    } catch (error) {
        processStatus.textContent = "ไม่สามารถเชื่อมต่อ Backend ได้";
    } finally {
        evaluateButton.disabled = false;
    }
});
