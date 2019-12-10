
import java.util.Properties;

import javax.mail.Authenticator;
import javax.mail.Message;
import javax.mail.MessagingException;
import javax.mail.PasswordAuthentication;
import javax.mail.Session;
import javax.mail.Transport;
import javax.mail.internet.InternetAddress;
import javax.mail.internet.MimeMessage;

public class JavaMailUtil {
	public static void sendMail(String recepient, String user, String time, String date, boolean okay) throws MessagingException{
		System.out.println("Preparing to send email");
		Properties properties = new Properties();
		
		properties.put("mail.smtp.auth", "true");
		properties.put("mail.smtp.starttls.enable", "true");
		properties.put("mail.smtp.host", "smtp.gmail.com");
		properties.put("mail.smtp.port", "587");

		String myAccountEmail = "wirelessinhomefalldetection@gmail.com";
		String myAccountPassword = "WIFU2019";
		
		Session session = Session.getInstance(properties, new Authenticator() {
			@Override
			protected PasswordAuthentication getPasswordAuthentication() {
				return new PasswordAuthentication(myAccountEmail, myAccountPassword);
			}
		});
		
		Message message = prepareMessage(session, myAccountEmail, recepient, user, time, date, okay);
		
		Transport.send(message);
		System.out.println("Message sent successfully");	
	}
	
	private static Message prepareMessage(Session session, String myAccountEmail, String recepient, String user, String time, String date, boolean okay) {
			Message message = new MimeMessage(session);
			try {
				if(okay == false)
                                {
                                    message.setFrom(new InternetAddress(myAccountEmail));
                                    message.setRecipient(Message.RecipientType.TO, new InternetAddress(recepient));
                                    message.setSubject("A Fall Has Been Detected");
                                    message.setText("You are a caregiver for " + user + " and it was detected that a fall has occurred at " 
                                                     + time + " on " + date + ".");
                                    return message;
                                }
                                else if(okay == true)
                                {
                                    message.setFrom(new InternetAddress(myAccountEmail));
                                    message.setRecipient(Message.RecipientType.TO, new InternetAddress(recepient));
                                    message.setSubject("A Fall Has Been Detected");
                                    message.setText("You are a caregiver for " + user + " and they have said they are OKAY after the fall that occurred at " 
                                                     + time + " on " + date + ".");
                                    return message;
                                }
			} catch (MessagingException e) {
				e.printStackTrace();
			}
			return null;
	
	}

}
