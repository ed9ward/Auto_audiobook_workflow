// Foxy AudioBook Maker — запускатель для Windows (C#, встроенный в Windows .NET Framework 4.8).
// Внутри exe лежит audiobook_master.py. Запускатель находит Python 3.14 (если его нет — с согласия пользователя скачивает
// официальный установщик с python.org и ставит только для текущей учётной записи), создаёт рабочую папку .venv рядом с exe,
// при первом запуске ставит библиотеки окна и звука, показывая ход работы, и запускает программу без чёрной консоли.
// Сборка: csc /target:winexe /win32icon:launcher\icon.ico /resource:audiobook_master.py,audiobook_master.py launcher\FoxyAudioBookMaker.cs
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Drawing;
using System.Globalization;
using System.IO;
using System.Net;
using System.Reflection;
using System.Security.Cryptography;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading;
using System.Windows.Forms;

static class Launcher
{
    const string Title = "Foxy AudioBook Maker";
    const string ScriptName = "audiobook_master.py";
    const string PythonVersion = "3.14.10";               // у 3.12 и 3.13 установщиков для Windows больше нет: берём 3.14
    const string PythonUrl = "https://www.python.org/ftp/python/3.14.10/python-3.14.10-amd64.exe";
    const string PythonSha256 = "92480239264744fe0386a1bd8275786fefa126e3129e56a68e22b52c73329965";   // SHA-256 со страницы выпуска на python.org
    const string ReadyMarker = "audiobook-ready.txt";         // лежит в .venv: первый запуск уже подготовлен
    const int NoTk = 2;                                       // код выхода программы при подготовке: в Python нет Tk

    static string Base;
    static bool Ru;

    [STAThread]
    static int Main(string[] args)
    {
        Base = Path.GetDirectoryName(Path.GetFullPath(Application.ExecutablePath));
        Ru = DetectRussian();
        if (args.Length > 0 && args[0] == "--launcher-selftest")
            return SelfTest();
        Application.EnableVisualStyles();
        Application.SetCompatibleTextRenderingDefault(false);
        if (!AcquireInstanceMutex())
        {
            Error(T("Программа уже запускается из этой папки. Подождите немного или закройте другое окно.",
                    "The program is already starting from this folder. Wait a moment or close the other window."));
            return 0;
        }
        try
        {
            return Run(args);
        }
        catch (Exception e)
        {
            Error(T("Не удалось запустить программу:\n\n", "Could not start the program:\n\n") + e.Message);
            return 1;
        }
        finally
        {
            try { instanceMutex.ReleaseMutex(); } catch (Exception) { }
        }
    }

    static Mutex instanceMutex;                                   // держим, пока запускатель готовит окружение и запускает программу

    // Два запускателя из одной папки не должны одновременно ставить Python и готовить .venv. Ждём до 5 секунд: после
    // обновления новый exe запускается, пока прежний ещё закрывается.
    static bool AcquireInstanceMutex()
    {
        string key;
        using (var md5 = MD5.Create())
            key = BitConverter.ToString(md5.ComputeHash(Encoding.UTF8.GetBytes(Base.ToLowerInvariant()))).Replace("-", "");
        instanceMutex = new Mutex(false, "Local\\FoxyAudioBookMaker-" + key);
        try
        {
            return instanceMutex.WaitOne(5000);
        }
        catch (AbandonedMutexException)
        {
            return true;                                          // прежний запускатель завершился аварийно: мьютекс теперь наш
        }
    }

    static int Run(string[] args)
    {
        string venv = Path.Combine(Base, ".venv");
        string venvPython = Path.Combine(venv, "Scripts", "python.exe");
        if (!File.Exists(venvPython))
        {
            string python = FindPython();
            if (python == null)
            {
                string folder = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
                                             "Programs", "Python", "Python314");
                if (!Ask(string.Format(T(
                        "Для работы нужен Python 3.14, а на этом компьютере его нет.\n\nПрограмма скачает официальный " +
                        "установщик с python.org (около 26 МБ) и установит Python {0} только для вашей учётной записи, " +
                        "без прав администратора:\n{1}\n\nПродолжить?",
                        "Python 3.14 is needed but is not installed on this computer.\n\nThe program will download the " +
                        "official installer from python.org (about 26 MB) and install Python {0} for your account only, " +
                        "without administrator rights:\n{1}\n\nContinue?"), PythonVersion, folder)))
                    return 1;
                if (!InstallPython())
                    return 1;
                python = FindPython();
                if (python == null)
                {
                    Error(T("Python установился, но программа его не нашла. Перезагрузите компьютер и запустите программу снова.",
                            "Python was installed, but the program cannot find it. Restart the computer and try again."));
                    return 1;
                }
            }
            if (!Ask(string.Format(T(
                    "Создать рабочую папку программы здесь?\n\n{0}\n\nВ неё ставятся библиотеки: для Gemini около 300 МБ, " +
                    "для Qwen около 5 ГБ (Qwen ставится позже, только если вы его выберете).",
                    "Create the program's working folder here?\n\n{0}\n\nLibraries are installed into it: about 300 MB for " +
                    "Gemini, about 5 GB for Qwen (Qwen is installed later, only if you choose it)."), venv)))
                return 1;
            int created = new StepWindow(T("Создаю рабочую папку .venv…", "Creating the .venv working folder…"))
                .RunProcess(python, "-m venv " + Quote(venv), Env());
            if (created != 0 || !File.Exists(venvPython))
                return 1;
        }

        string script = ExtractScript();
        if (!File.Exists(Path.Combine(venv, ReadyMarker)))
        {
            int prepared = new StepWindow(T("Первый запуск: ставлю библиотеки окна и звука. Это займёт пару минут…",
                                            "First start: installing the window and audio libraries. This takes a couple of minutes…"))
                .RunProcess(venvPython, Quote(script) + " --prepare", Env());
            if (prepared == NoTk)
            {
                if (!Ask(T("В этом Python нет Tk (библиотеки окна), поэтому окно открыть нельзя. Как это исправить, " +
                           "написано в журнале выше.\n\nОткрыть программу в текстовом режиме?",
                           "This Python has no Tk (the window library), so the window cannot open. How to fix it is " +
                           "shown in the log above.\n\nOpen the program in text mode?")))
                    return 1;
                return Start(venvPython, script, new[] { "--console" }, true);
            }
            if (prepared != 0)
                return 1;
        }
        bool console = Array.Exists(args, a => a == "--console" || a == "--check" || a == "--auto");
        return Start(console ? venvPython : Path.Combine(venv, "Scripts", "pythonw.exe"), script, args, console);
    }

    static int Start(string python, string script, string[] args, bool console)
    {
        var info = new ProcessStartInfo(python, Quote(script) + JoinArgs(args));
        info.UseShellExecute = false;
        info.WorkingDirectory = Base;
        info.CreateNoWindow = !console;
        foreach (var pair in Env())
            info.EnvironmentVariables[pair.Key] = pair.Value;
        Process.Start(info);
        return 0;
    }

    static Dictionary<string, string> Env()
    {
        var env = new Dictionary<string, string>();
        env["AUDIOBOOK_HOME"] = Base;                        // все папки программы — рядом с exe, как и раньше
        env["AUDIOBOOK_EXE"] = Path.GetFullPath(Application.ExecutablePath);
        env["PYTHONIOENCODING"] = "utf-8";
        env["PYTHONUTF8"] = "1";
        return env;
    }

    // ---- код программы, вшитый в exe
    static byte[] ScriptBytes()
    {
        using (Stream stream = Assembly.GetExecutingAssembly().GetManifestResourceStream(ScriptName))
        {
            if (stream == null)
                throw new InvalidOperationException("audiobook_master.py is not embedded");
            var memory = new MemoryStream();
            stream.CopyTo(memory);
            return memory.ToArray();
        }
    }

    static string ExtractScript()
    {
        string folder = Path.Combine(Base, ".cache", "app");
        Directory.CreateDirectory(folder);
        string path = Path.Combine(folder, ScriptName);
        byte[] bytes = ScriptBytes();
        if (!File.Exists(path) || !SameBytes(File.ReadAllBytes(path), bytes))
            File.WriteAllBytes(path, bytes);
        return path;
    }

    static bool SameBytes(byte[] a, byte[] b)
    {
        if (a.Length != b.Length)
            return false;
        for (int i = 0; i < a.Length; i++)
            if (a[i] != b[i])
                return false;
        return true;
    }

    static int SelfTest()
    {
        byte[] bytes = ScriptBytes();
        string text = Encoding.UTF8.GetString(bytes);
        return text.Contains("APP_TITLE = \"Foxy AudioBook Maker\"") ? 0 : 3;
    }

    // ---- Python: найти или поставить
    static string FindPython()
    {
        foreach (string version in new[] { "3.14" })
        {
            string found = Capture("py", "-" + version + " -c \"import sys; print(sys.executable)\"");
            if (found != null && File.Exists(found.Trim()))
                return found.Trim();
            string local = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
                                        "Programs", "Python", "Python" + version.Replace(".", ""), "python.exe");
            if (File.Exists(local))
                return local;
        }
        return null;
    }

    static string Capture(string file, string arguments)
    {
        try
        {
            var info = new ProcessStartInfo(file, arguments);
            info.UseShellExecute = false;
            info.CreateNoWindow = true;
            info.RedirectStandardOutput = true;
            using (Process process = Process.Start(info))
            {
                string output = process.StandardOutput.ReadToEnd();
                process.WaitForExit(30000);
                return process.ExitCode == 0 ? output : null;
            }
        }
        catch (Exception)
        {
            return null;                                     // нет py или он не нашёл нужную версию
        }
    }

    static bool InstallPython()
    {
        string installer = Path.Combine(Path.GetTempPath(), "python-" + PythonVersion + "-amd64.exe");
        var window = new StepWindow(T("Скачиваю установщик Python с python.org…", "Downloading the Python installer from python.org…"));
        if (!window.Download(PythonUrl, installer))
            return false;
        if (Sha256(installer) != PythonSha256)
        {
            File.Delete(installer);
            Error(T("Скачанный установщик повреждён (контрольная сумма не совпала). Попробуйте ещё раз.",
                    "The downloaded installer is damaged (checksum mismatch). Please try again."));
            return false;
        }
        int code = new StepWindow(T("Устанавливаю Python 3.14. Это займёт минуту-две…", "Installing Python 3.14. This takes a minute or two…"))
            .RunProcess(installer, "/quiet InstallAllUsers=0 Include_tcltk=1 Include_launcher=1 InstallLauncherAllUsers=0 " +
                        "PrependPath=0 Include_test=0 Include_doc=0 Shortcuts=0", null);
        try { File.Delete(installer); } catch (Exception) { }
        if (code != 0 && code != 3010)
        {
            Error(string.Format(T("Python установить не удалось (код {0}). Установите Python 3.14 вручную с python.org и запустите программу снова.",
                                  "Python could not be installed (code {0}). Install Python 3.14 from python.org by hand and start the program again."), code));
            return false;
        }
        return true;
    }

    static string Sha256(string path)
    {
        using (var sha = SHA256.Create())
        using (var stream = File.OpenRead(path))
            return BitConverter.ToString(sha.ComputeHash(stream)).Replace("-", "").ToLowerInvariant();
    }

    // ---- язык и мелочи
    static bool DetectRussian()
    {
        try
        {
            string settings = Path.Combine(Base, "settings.json");
            if (File.Exists(settings))
            {
                Match m = Regex.Match(File.ReadAllText(settings, Encoding.UTF8), "\"language\"\\s*:\\s*\"(\\w+)\"");
                if (m.Success && (m.Groups[1].Value == "ru" || m.Groups[1].Value == "en"))
                    return m.Groups[1].Value == "ru";
            }
        }
        catch (Exception) { }
        return CultureInfo.CurrentUICulture.TwoLetterISOLanguageName == "ru";
    }

    internal static string T(string ru, string en) { return Ru ? ru : en; }

    static bool Ask(string text)
    {
        return MessageBox.Show(text, Title, MessageBoxButtons.OKCancel, MessageBoxIcon.Question) == DialogResult.OK;
    }

    internal static void Error(string text)
    {
        MessageBox.Show(text, Title, MessageBoxButtons.OK, MessageBoxIcon.Error);
    }

    static string Quote(string value) { return "\"" + value + "\""; }

    static string JoinArgs(string[] args)
    {
        var text = new StringBuilder();
        foreach (string arg in args)
            text.Append(" ").Append(Quote(arg));
        return text.ToString();
    }

    internal static Icon AppIcon()
    {
        try { return Icon.ExtractAssociatedIcon(Application.ExecutablePath); } catch (Exception) { return null; }
    }
}

// Окно одного шага: подпись, полоса прогресса, журнал и кнопка «Отмена». Пока шаг идёт, пользователь видит, что происходит.
class StepWindow : Form
{
    readonly Label caption = new Label();
    readonly ProgressBar bar = new ProgressBar();
    readonly TextBox journal = new TextBox();
    readonly Button cancel = new Button();
    volatile bool cancelled;
    Process process;
    int result = -1;

    public StepWindow(string title)
    {
        Text = "Foxy AudioBook Maker";
        Icon = Launcher.AppIcon();
        FormBorderStyle = FormBorderStyle.FixedDialog;
        MaximizeBox = MinimizeBox = false;
        StartPosition = FormStartPosition.CenterScreen;
        ClientSize = new Size(560, 300);
        Font = new Font("Segoe UI", 9.5f);
        caption.Text = title;
        caption.SetBounds(16, 14, 528, 40);
        bar.SetBounds(16, 58, 528, 16);
        bar.Style = ProgressBarStyle.Marquee;
        journal.SetBounds(16, 86, 528, 162);
        journal.Multiline = true;
        journal.ReadOnly = true;
        journal.ScrollBars = ScrollBars.Vertical;
        journal.Font = new Font("Consolas", 8.5f);
        cancel.Text = Launcher.T("Отмена", "Cancel");
        cancel.SetBounds(444, 258, 100, 28);
        cancel.Click += delegate { cancelled = true; Kill(); Close(); };
        Controls.AddRange(new Control[] { caption, bar, journal, cancel });
    }

    void Log(string line)
    {
        if (line == null)
            return;
        BeginInvoke((MethodInvoker)delegate
        {
            journal.AppendText(line + Environment.NewLine);
            string[] lines = journal.Lines;
            if (lines.Length > 400)                           // журнал ограничен: держим последние 300 строк
            {
                string[] tail = new string[300];
                Array.Copy(lines, lines.Length - 300, tail, 0, 300);
                journal.Lines = tail;
                journal.SelectionStart = journal.TextLength;
                journal.ScrollToCaret();
            }
        });
    }

    void Kill()
    {
        try
        {
            if (process != null && !process.HasExited)
            {
                process.Kill();
                process.WaitForExit(5000);                   // дождаться, пока процесс действительно остановится
            }
        }
        catch (Exception) { }
    }

    // Запускает программу, показывая её вывод в журнале. Возвращает код выхода (-1 — отменено). При ошибке окно
    // остаётся открытым с журналом, чтобы было видно, что случилось.
    public int RunProcess(string file, string arguments, Dictionary<string, string> env)
    {
        Shown += delegate
        {
            new Thread(delegate ()
            {
                try
                {
                    var info = new ProcessStartInfo(file, arguments);
                    info.UseShellExecute = false;
                    info.CreateNoWindow = true;
                    info.RedirectStandardOutput = info.RedirectStandardError = true;
                    info.StandardOutputEncoding = info.StandardErrorEncoding = Encoding.UTF8;
                    if (env != null)
                        foreach (var pair in env)
                            info.EnvironmentVariables[pair.Key] = pair.Value;
                    if (cancelled)                                // «Отмена» нажали ещё до запуска
                        return;
                    process = Process.Start(info);
                    if (cancelled)                                // нажали, пока процесс запускался: останавливаем его
                    {
                        Kill();
                        return;
                    }
                    process.OutputDataReceived += delegate (object s, DataReceivedEventArgs e) { Log(e.Data); };
                    process.ErrorDataReceived += delegate (object s, DataReceivedEventArgs e) { Log(e.Data); };
                    process.BeginOutputReadLine();
                    process.BeginErrorReadLine();
                    process.WaitForExit();
                    result = process.ExitCode;
                }
                catch (Exception e)
                {
                    Log(e.Message);
                    result = 1;
                }
                BeginInvoke((MethodInvoker)delegate { Finish(); });
            }) { IsBackground = true }.Start();
        };
        Application.Run(this);
        return cancelled ? -1 : result;
    }

    // Скачивает файл с полосой прогресса. False — ошибка или отмена.
    public bool Download(string url, string target)
    {
        bool ok = false;
        ServicePointManager.SecurityProtocol |= SecurityProtocolType.Tls12;
        var client = new WebClient();
        bar.Style = ProgressBarStyle.Continuous;
        client.DownloadProgressChanged += delegate (object s, DownloadProgressChangedEventArgs e)
        {
            bar.Value = Math.Min(100, e.ProgressPercentage);
            caption.Text = string.Format(Launcher.T("Скачиваю установщик Python с python.org… {0} из {1} МБ",
                                                    "Downloading the Python installer from python.org… {0} of {1} MB"),
                                         e.BytesReceived / 1048576, Math.Max(1, e.TotalBytesToReceive / 1048576));
        };
        client.DownloadFileCompleted += delegate (object s, System.ComponentModel.AsyncCompletedEventArgs e)
        {
            if (e.Cancelled || cancelled)
                return;
            if (e.Error != null)
            {
                Log(e.Error.Message);
                result = 1;
                Finish();
                return;
            }
            ok = true;
            Close();
        };
        cancel.Click += delegate { client.CancelAsync(); };
        Shown += delegate { client.DownloadFileAsync(new Uri(url), target); };
        Application.Run(this);
        return ok && !cancelled;
    }

    void Finish()
    {
        if (result == 0)
        {
            Close();
            return;
        }
        bar.Style = ProgressBarStyle.Continuous;
        bar.Value = 0;
        caption.Text = Launcher.T("Не получилось. Что случилось, видно в журнале ниже.",
                                  "That did not work. The log below shows what happened.");
        cancel.Text = Launcher.T("Закрыть", "Close");
    }
}
