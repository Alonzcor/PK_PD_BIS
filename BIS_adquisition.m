%% T is the raw data of time, EEG and BIS
T = readtable("paciente_1_BIS.csv");
head(T); % --> Theres 'NaN' data, WE NEED TO DELETE IT

%% Filter to delete 'NaN' values
validos = ~isnan(T.EEG_uV) & ~isnan(T.BIS_real) & (abs(T.EEG_uV) <= 150);
T_clean = T(validos, :); % (rows, columns)
head(T_clean);

%% Vector extraction
t_min = T_clean.Tiempo_s / 60; %convert seconds into minutes
eeg = T_clean.EEG_uV;
bis = T_clean.BIS_real;

%% First Graph
plot_eeg_bis(t_min, eeg, bis, 'EEG first filter vs BIS measured', [min(t_min) max(t_min)], [-150 150]);
plot_eeg_bis(t_min, eeg, bis, 'EEG first filter vs BIS measured', [80.0 81.0], [-150 150]);

%BIG PROBLEM, eeg isn't filtered enough

%% Bandpass Filter (0.5 - 45 Hz) at fs = 128 Hz
fs = 128;              % Sampling frequency (Hz)
lowcut = 0.5;          % High-pass cutoff (removes DC offset & slow drift)
highcut = 45;          % Low-pass cutoff (removes high-frequency EMG/line noise)

[b, a] = butter(4, [lowcut highcut] / (fs / 2), 'bandpass'); %butterworth filtering
eeg_filtered = filtfilt(b, a, eeg); % Zero-phase filtering

%% Graph wit bandpass
plot_eeg_bis(t_min, eeg_filtered, bis, 'EEG filter by freq. vs BIS measured', [min(t_min) max(t_min)], [-200 200])
plot_eeg_bis(t_min, eeg_filtered, bis, 'EEG filter by freq. vs BIS measured', [80.0 81.0], [-200 200])


%% ==================== EXTRACCIÓN DE LOS 3 PARÁMETROS DEL BIS ====================
fs = 128;
win_sec = 4;                     % Ventanas de 4 segundos
win_len = fs * win_sec;          % 512 muestras por ventana
n_win = floor(length(eeg_filtered) / win_len);

% Vectores para guardar resultados por ventana
t_win     = zeros(n_win, 1);
bis_win   = zeros(n_win, 1);
beta_rat  = zeros(n_win, 1);
synch_fs  = zeros(n_win, 1);
bsr_val   = zeros(n_win, 1);

% Vector de frecuencias para ventanas de 4 s (resolución = 128/512 = 0.25 Hz)
f_win = (0:win_len-1)' * (fs / win_len);
idx_fast = (f_win >= 30) & (f_win <= 47);
idx_slow = (f_win >= 11) & (f_win <= 20);

% Índices de frecuencia para el biespectro (hasta 47 Hz)
idx_all_b  = find(f_win >= 0.5 & f_win <= 47);
idx_fast_b = find(f_win >= 40  & f_win <= 47);

for k = 1:n_win
    i1 = (k - 1) * win_len + 1;
    i2 = k * win_len;
    
    seg = eeg_filtered(i1:i2);
    t_win(k)   = mean(t_min(i1:i2));
    bis_win(k) = mean(bis(i1:i2));
    
    % --- 1. BETA RATIO (Dominio de Frecuencia) ---
    X = fft(seg);
    P = abs(X).^2; % Densidad espectral de potencia
    P_fast = sum(P(idx_fast));
    P_slow = sum(P(idx_slow)) + eps; % +eps evita división por cero
    beta_rat(k) = log10(P_fast / P_slow);
    
    % --- 2. SYNCHFASTSLOW (Dominio Biespectral) ---
    % Calculamos el acoplamiento B(f1, f2) = |X(f1) * X(f2) * conj(X(f1+f2))|
    B_all = 0;
    for a = 1:length(idx_all_b)
        for b = a:length(idx_all_b)
            f1_i = idx_all_b(a);
            f2_i = idx_all_b(b);
            fsum_i = f1_i + f2_i - 1; % Índice de f1 + f2
            if fsum_i <= length(X)
                B_all = B_all + abs(X(f1_i) * X(f2_i) * conj(X(fsum_i)));
            end
        end
    end
    
    B_fast = 0;
    for a = 1:length(idx_fast_b)
        for b = a:length(idx_fast_b)
            f1_i = idx_fast_b(a);
            f2_i = idx_fast_b(b);
            fsum_i = f1_i + f2_i - 1;
            if fsum_i <= length(X)
                B_fast = B_fast + abs(X(f1_i) * X(f2_i) * conj(X(fsum_i)));
            end
        end
    end
    synch_fs(k) = log10((B_fast + eps) / (B_all + eps));
    
    % --- 3. BSR - BURST SUPPRESSION RATIO (Dominio del Tiempo) ---
    % Criterio oficial: |EEG| < 5 uV durante al menos 0.5 s (64 muestras seguidas)
    es_plano = abs(seg) < 5;
    % Detectar rachas de 64 muestras consecutivas en silencio usando convolución
    rachas = conv(double(es_plano), ones(64, 1), 'same') >= 64;
    bsr_val(k) = (sum(rachas) / win_len) * 100;
end

% Suavizado móvil (ventana de 8 bloques = 32 s, similar al monitor real)
beta_rat_s = movmean(beta_rat, 8);
synch_fs_s = movmean(synch_fs, 8);
bsr_val_s  = movmean(bsr_val,  8);

%% ==================== VERIFICACIÓN EN TERMINAL ====================
despierto = bis_win > 80;
cirugia   = (bis_win >= 40) & (bis_win <= 60);
profundo  = bis_win < 35;

fprintf('\n=================================================================================\n');
fprintf(' VERIFICACION DE PARAMETROS EXTRAIDOS EN PACIENTE 1 (VALORES PROMEDIO)\n');
fprintf('=================================================================================\n');
fprintf(' Parametro      | Despierto (BIS>80) | Cirugia (BIS 40-60) | Profundo (BIS<35)\n');
fprintf('---------------------------------------------------------------------------------\n');
fprintf(' BetaRatio      |      %+6.3f        |       %+6.3f        |      %+6.3f\n', ...
        mean(beta_rat_s(despierto)), mean(beta_rat_s(cirugia)), mean(beta_rat_s(profundo)));
fprintf(' SynchFastSlow  |      %+6.3f        |       %+6.3f        |      %+6.3f\n', ...
        mean(synch_fs_s(despierto)), mean(synch_fs_s(cirugia)), mean(synch_fs_s(profundo)));
fprintf(' BSR (Silencio) |      %5.1f %%       |       %5.1f %%       |      %5.1f %%\n', ...
        mean(bsr_val_s(despierto)), mean(bsr_val_s(cirugia)), mean(bsr_val_s(profundo)));
fprintf('=================================================================================\n\n');

%% ==================== GRÁFICAS DE VERIFICACIÓN ====================
figure('Name', 'Los 3 Parametros del BIS vs BIS Real');

ax1 = subplot(4,1,1);
plot(t_win, beta_rat_s, 'LineWidth', 1.2);
title('1. BetaRatio (Velocidad / Frecuencia) -> Cae cuando el paciente se duerme');
ylabel('log10 Ratio'); grid on;

ax2 = subplot(4,1,2);
plot(t_win, synch_fs_s, 'LineWidth', 1.2);
title('2. SynchFastSlow (Sincronia de Fase) -> Modula la zona de cirugia (BIS 40-60)');
ylabel('log10 Biesp.'); grid on;

ax3 = subplot(4,1,3);
plot(t_win, bsr_val_s, 'LineWidth', 1.2);
title('3. BSR (Tiempo Muerto) -> Sube en los tramos profundos (min 10-38 y 75-90)');
ylabel('Silencio (%)'); ylim([-5 100]); grid on;

ax4 = subplot(4,1,4);
plot(t_win, bis_win, 'LineWidth', 1.5); hold on;
yline(60, '--g'); yline(40, '--r');
title('4. BIS Real del Monitor');
xlabel('Time (minutes)'); ylabel('BIS (0-100)'); ylim([0 100]); grid on;

linkaxes([ax1, ax2, ax3, ax4], 'x');
%% ==================== FUNCIÓN LOCAL (VA HASTA ABAJO) ====================
function plot_eeg_bis(t_min, eeg_sig, bis_sig, titulo_eeg, rango_t, limite_y)
    % Valores por defecto si no envías rango de tiempo o límite de amplitud
    if nargin < 5 || isempty(rango_t)
        rango_t = [min(t_min) max(t_min)];
    end
    if nargin < 6 || isempty(limite_y)
        limite_y = [-150 150];
    end

    figure('Name', titulo_eeg);

    % Panel 1: Señal EEG (raw, filtered, etc.)
    ax1 = subplot(2, 1, 1);
    plot(t_min, eeg_sig);
    title(titulo_eeg);
    ylabel('Amplitude (\muV)');
    ylim(limite_y);
    xlim(rango_t);
    grid on;

    % Panel 2: Evolución BIS
    ax2 = subplot(2, 1, 2);
    plot(t_min, bis_sig);
    title('BIS evolution');
    xlabel('Time (minutes)');
    ylabel('BIS (0-100)');
    ylim([-1 100]);
    xlim(rango_t);
    grid on;

    % Sincroniza ambos ejes X si haces zoom manual
    linkaxes([ax1, ax2], 'x');
end